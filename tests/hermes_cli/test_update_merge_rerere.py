"""Real Git regression for custom-branch update merge/rerere, no live services."""

from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hermes_cli import update_cmd


class RerereUpdateMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "rerere.enabled", "true")
        self.git("config", "rerere.autoupdate", "true")
        self.file = self.repo / "example.txt"
        self.file.write_text("base\n")
        self.git("add", ".")
        self.git("commit", "-m", "base")
        self.git("switch", "-c", "overlay")
        self.file.write_text("local\n")
        self.git("commit", "-am", "local")
        self.before = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("switch", "main")
        self.file.write_text("upstream\n")
        self.git("commit", "-am", "upstream")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("switch", "overlay")

    def git(self, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=self.repo, text=True, capture_output=True, check=check
        )

    def train(self):
        self.assertEqual(self.git("merge", "origin/main", check=False).returncode, 1)
        self.file.write_text("reviewed combined resolution\n")
        self.git("rerere")
        self.git("merge", "--abort")

    def reconcile(self):
        with (
            patch.object(
                update_cmd, "_m", return_value=SimpleNamespace(PROJECT_ROOT=self.repo)
            ),
            patch.object(update_cmd, "_record_stop"),
        ):
            update_cmd._reconcile_diverged_checkout(["git"], "main", self.before)

    def test_cached_resolution_commits_without_discarding_either_parent(self):
        self.train()
        self.reconcile()
        self.assertEqual(self.file.read_text(), "reviewed combined resolution\n")
        parents = self.git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()
        self.assertEqual(len(parents), 3)
        self.assertEqual(parents[1], self.before)
        self.assertEqual(
            parents[2], self.git("rev-parse", "origin/main").stdout.strip()
        )
        self.assertEqual(self.git("status", "--porcelain").stdout, "")

    def test_unknown_conflict_still_aborts_and_preserves_local_head(self):
        with self.assertRaises(SystemExit):
            self.reconcile()
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.before)
        self.assertEqual(self.file.read_text(), "local\n")
        self.assertEqual(self.git("status", "--porcelain").stdout, "")

    def test_no_autoupdate_never_commits_unstaged_cached_resolution(self):
        self.train()
        self.git("config", "rerere.autoupdate", "false")
        with self.assertRaises(SystemExit):
            self.reconcile()
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.before)

    def test_commit_failure_is_not_accepted_as_update_success(self):
        self.train()
        real_git_run = update_cmd._git_run

        def fail_commit(command, args, **kwargs):
            if args[0] == "commit":
                return subprocess.CompletedProcess(
                    command + args, 7, "", "injected commit failure"
                )
            return real_git_run(command, args, **kwargs)

        with (
            patch.object(update_cmd, "_git_run", side_effect=fail_commit),
            self.assertRaises(SystemExit),
        ):
            self.reconcile()
        self.assertEqual(self.git("rev-parse", "HEAD").stdout.strip(), self.before)
        self.assertEqual(self.git("status", "--porcelain").stdout, "")


if __name__ == "__main__":
    unittest.main()

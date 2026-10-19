"""Finish a custom-branch merge only after opt-in rerere staged every conflict."""

from __future__ import annotations


def finish_rerere_merge(git_run, git_cmd) -> bool:
    """Git merge returns 1 even when rerere.autoupdate resolves all conflicts.

    Never treat a transport/merge failure or an unstaged resolution as a clean
    merge. The existing Git custody policy still owns the commit subprocess;
    a commit failure must stop the update just like an unresolved conflict.
    """
    configured = git_run(git_cmd, ["config", "--bool", "rerere.autoupdate"])
    if configured.returncode or configured.stdout.strip() != "true":
        return False
    if git_run(git_cmd, ["rev-parse", "--verify", "MERGE_HEAD"]).returncode:
        return False
    unmerged = git_run(git_cmd, ["ls-files", "--unmerged"])
    if unmerged.returncode or unmerged.stdout.strip():
        return False
    if git_run(git_cmd, ["diff", "--quiet"]).returncode:
        return False
    if git_run(git_cmd, ["diff", "--cached", "--check"]).returncode:
        return False
    return git_run(git_cmd, ["commit", "--no-edit"]).returncode == 0

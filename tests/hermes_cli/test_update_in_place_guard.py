from __future__ import annotations

import subprocess
from types import SimpleNamespace

from hermes_cli import update_cmd


def test_clean_in_place_update_skips_patch_equivalence_scan(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_git(_git_cmd, args, _cwd=None):
        calls.append(args)
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args == ["rev-parse", "--verify", "--quiet", "origin/main"]:
            return subprocess.CompletedProcess(args, 0, stdout="deadbeef\n", stderr="")
        raise AssertionError(f"unexpected git command: {args}")

    module = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        _assess_parked_branch_switch=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("in-place updates must not run git cherry")
        ),
    )
    monkeypatch.setattr(update_cmd, "_updates_config", lambda: {
        "parked_branch_strategy": "update_in_place"
    })
    monkeypatch.setattr(update_cmd, "_git_run", fake_git)
    monkeypatch.setattr(update_cmd, "_m", lambda: module)

    result = update_cmd._apply_parked_branch_guard(
        ["git"], "main", "hermesops-overlay",
        switch_branch=False, _windows_gateway_resume=None,
    )

    assert result == (False, True, "unmerged:unknown")
    assert calls == [
        ["status", "--porcelain"],
        ["rev-parse", "--verify", "--quiet", "origin/main"],
    ]

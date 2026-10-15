"""Cron directory creation distinguishes named profiles from profile-local scratch."""

import pytest

from cron.jobs import _ensure_cron_dir


@pytest.mark.parametrize("relative", ["cron", "cron/external-workers", "cron/output/job-1", "scripts"])
def test_custom_home_in_profile_scratch_creates_parents(tmp_path, relative):
    home = tmp_path / "profiles" / "silas" / "cache" / "scratch" / "custom-home"
    target = home / relative

    _ensure_cron_dir(target)

    assert target.is_dir()


@pytest.mark.parametrize("relative", ["cron", "cron/external-workers", "cron/output/job-1", "scripts"])
def test_deleted_named_profile_is_not_resurrected(tmp_path, relative):
    home = tmp_path / "profiles" / "deleted"
    home.mkdir(parents=True)
    home.rmdir()

    with pytest.raises(FileNotFoundError):
        _ensure_cron_dir(home / relative)

    assert not home.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("relative", ["cron", "scripts"])
def test_symlinked_named_profile_home_keeps_raw_path_guard(tmp_path, relative):
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    target_home = tmp_path / "external-home"
    target_home.mkdir()
    home = profiles / "linked"
    home.symlink_to(target_home, target_is_directory=True)
    target_home.rmdir()

    with pytest.raises(FileNotFoundError):
        _ensure_cron_dir(home / relative)

    assert home.is_symlink()
    assert not target_home.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("relative", ["cron", "scripts"])
def test_symlinked_parent_keeps_resolved_path_guard(tmp_path, relative):
    root = tmp_path / "actual"
    profiles = root / "profiles"
    home = profiles / "deleted"
    home.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(profiles, target_is_directory=True)
    home.rmdir()

    with pytest.raises(FileNotFoundError):
        _ensure_cron_dir(alias / "deleted" / relative)

    assert not home.exists()

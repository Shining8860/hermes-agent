"""The workspace and admission paths agree on scoped keys and disabled plugins."""
import pytest

from pm.package import InstallError
from pm.workspace import enabled_member_dirs


def test_scoped_members_respect_disabled_and_keep_other_profiles(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    current = home / "plugins" / "group" / "plugin"
    other_home = home / "profiles" / "other"
    other = other_home / "plugins" / "same"
    for member in (current, other):
        member.mkdir(parents=True)
        (member / "pyproject.toml").write_text('[project]\nname="test"\nversion="1"\n')
    (home / "config.yaml").write_text('plugins:\n  enabled: [group/plugin]\n  disabled: []\n')
    (other_home / "config.yaml").write_text('plugins:\n  enabled: [same]\n')
    assert set(enabled_member_dirs()) == {current, other}
    (home / "config.yaml").write_text('plugins:\n  enabled: [group/plugin]\n  disabled: [group/plugin]\n')
    assert enabled_member_dirs() == [other]


def test_manifest_only_member_is_named_after_its_plugin_dir_and_stays_unique(tmp_path):
    """uv's conflict text names the workspace member (``hermes-plugin-<key> depends on …``);
    a bare path hash left the user with nothing to disable. Two same-named plugins from
    different homes must still be distinct members."""
    import tomllib
    from pm.workspace import _workspace_member

    members = []
    for home in ("home-a", "home-b"):
        plugin = tmp_path / home / "plugins" / "My Plugin!"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: my-plugin\npip_dependencies: [left-pad-py]\n", encoding="utf-8")
        root = tmp_path / f"gen-{home}"
        root.mkdir()
        members.append(_workspace_member(plugin, root, identity=plugin))
    names = [tomllib.loads((m / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"] for m in members]
    assert all(name.startswith("hermes-plugin-my-plugin-") for name in names), names
    assert names[0] != names[1]
    assert members[0].name != members[1].name


def test_pyproject_member_is_renamed_by_its_key_and_stays_unique(tmp_path):
    """A plugin that ships its own pyproject declares its real [project].name, and
    uv identifies workspace members by that name — enabling the plugin in two
    profiles then fails ``uv lock`` with "Two workspace members are both named
    …". Rename metadata-only members by their unique key, like manifest-only
    ones; a buildable member keeps the name its package metadata reports."""
    import tomllib
    from pm.workspace import _workspace_member

    members = []
    for home in ("home-a", "home-b"):
        plugin = tmp_path / home / "plugins" / "hindsight"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname = "hermes-plugin-hindsight"\nversion = "1.0.0"\n'
            'dependencies = ["hindsight-client>=0.10.1"]\n',
            encoding="utf-8",
        )
        root = tmp_path / f"gen-{home}"
        root.mkdir()
        members.append(_workspace_member(plugin, root, identity=plugin))
    names = [tomllib.loads((m / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"] for m in members]
    assert all(name.startswith("hermes-plugin-hindsight-") for name in names), names
    assert names[0] != names[1]
    document = tomllib.loads((members[0] / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["version"] == "1.0.0"
    assert document["project"]["dependencies"] == ["hindsight-client>=0.10.1"]


def test_tooling_only_pyproject_becomes_a_valid_virtual_member(tmp_path):
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "tooling-only"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[tool.ruff]\ntarget-version = "py311"\n', encoding="utf-8"
    )
    root = tmp_path / "generation"
    root.mkdir()

    member = _workspace_member(plugin, root, identity=plugin)
    document = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))
    assert document["project"]["name"].startswith("hermes-plugin-tooling-only-")
    assert document["project"]["version"] == "0.0.0"
    assert document["project"]["requires-python"] == ">=3.11"
    assert document["tool"]["ruff"]["target-version"] == "py311"


def test_buildable_pyproject_member_keeps_its_declared_name(tmp_path):
    """uv verifies a buildable member's [project].name against the package metadata
    its backend produces, so renaming it breaks the build ("Package metadata name
    … does not match given name"); only metadata-only members may be renamed."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "replay"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "replay-plugin"\nversion = "1.0"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    assert (member / "pyproject.toml").read_text(encoding="utf-8") == (
        plugin / "pyproject.toml").read_text(encoding="utf-8")


def test_identical_buildable_members_across_profiles_are_deduplicated(tmp_path):
    import tomllib
    from pm.workspace import _generate_pyproject

    source = tmp_path / "core"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "1"\n', encoding="utf-8"
    )
    plugins = []
    project = (
        '[project]\nname = "same_plugin"\nversion = "1"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n'
    )
    for home in ("a", "b"):
        plugin = tmp_path / home / "plugins" / "same"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(project, encoding="utf-8")
        (plugin / "backend.py").write_text("VALUE = 1\n", encoding="utf-8")
        plugins.append(plugin)

    root = tmp_path / "generation"
    _generate_pyproject(plugins, root, source=source)
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert len(document["tool"]["uv"]["workspace"]["members"]) == 1


@pytest.mark.parametrize("cache", [".pytest_cache", ".mypy_cache", ".ruff_cache", ".hypothesis"])
def test_tool_caches_do_not_change_member_identity_or_snapshot(tmp_path, cache):
    from pm.workspace import _deduplicate_buildable_members, _workspace_member, members_stamp

    plugins = []
    for home in ("a", "b"):
        plugin = tmp_path / home / "plugins" / "same"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname = "same-plugin"\nversion = "1"\n'
            '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
            encoding="utf-8",
        )
        (plugin / "backend.py").write_text("VALUE = 1\n", encoding="utf-8")
        plugins.append(plugin)

    before = members_stamp(plugins)
    for parent in (plugins[0], plugins[0] / "nested"):
        generated = parent / cache
        generated.mkdir(parents=True)
        (generated / "state").write_text("profile-local cache", encoding="utf-8")

    assert members_stamp(plugins) == before
    assert len(_deduplicate_buildable_members(dict(zip(plugins, plugins)))) == 1
    root = tmp_path / "generation"
    root.mkdir()
    member = _workspace_member(plugins[0], root, identity=plugins[0])
    assert not list(member.rglob(cache))
    assert (member / "backend.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.parametrize("package_mode", ["backend", "uv-package"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("staged", [False, True])
def test_seed_lock_keeps_existing_deduplicated_buildable_member(tmp_path, package_mode, reverse, staged):
    """Adding an earlier profile must not strand a seed lock's editable source."""
    import tomllib
    import shutil
    from unittest.mock import Mock
    from pm.environment import PythonEnvironment
    from pm.workspace import _member_key, lock_and_sync

    source = tmp_path / "core"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "1"\n', encoding="utf-8"
    )
    plugins = []
    for home in ("a", "b"):
        plugin = tmp_path / home / "plugins" / "same"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname = "same-plugin"\nversion = "1"\n'
            + ('[build-system]\nrequires = []\nbuild-backend = "backend"\n'
               if package_mode == "backend" else '[tool.uv]\npackage = true\n'),
            encoding="utf-8",
        )
        plugins.append(plugin)

    retained = plugins[0] if reverse else plugins[1]
    expected = f"plugin-sources/{_member_key(retained)}"
    seed = tmp_path / "uv.lock"
    seed_text = (
        'version = 1\n[[package]]\nname = "same-plugin"\nversion = "1"\n'
        f'source = {{ editable = "{expected}" }}\n'
    )
    seed.write_text(seed_text, encoding="utf-8")

    def check_sync(root, *, extras, frozen):
        document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        assert document["tool"]["uv"]["workspace"]["members"] == [expected]
        assert (root / expected / "pyproject.toml").is_file()
        assert (root / "uv.lock").read_text(encoding="utf-8") == seed_text

    entries = list(reversed(plugins)) if reverse else plugins
    if staged:
        staging = tmp_path / "staged-update"
        shutil.copytree(retained, staging)
        entries = {identity: staging if identity == retained else identity for identity in entries}
    environment = Mock(spec=PythonEnvironment)
    environment.sync.side_effect = check_sync
    lock_and_sync(entries, [], root=tmp_path / "generation", source=source,
                  seed_lock=seed, environment=environment)
    environment.sync.assert_called_once()
    assert seed.read_text(encoding="utf-8") == seed_text


def test_real_uv_seed_survives_adding_an_earlier_identical_profile(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    import tomllib
    from pathlib import Path

    from pm.environment import PythonEnvironment
    from pm.workspace import _member_key, lock_and_sync
    from tests.pm.test_workspace_build_inputs import _buildable_source

    source = tmp_path / "core"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "1"\nrequires-python = ">=3.11"\n'
        '[tool.uv]\npackage = false\nno-index = true\n', encoding="utf-8",
    )
    earlier = tmp_path / "a" / "plugins" / "same"
    retained = tmp_path / "b" / "plugins" / "same"
    retained.mkdir(parents=True)
    _buildable_source(retained)
    shutil.copytree(retained, earlier)
    cache = earlier / ".pytest_cache"
    cache.mkdir()
    (cache / "state").write_text("local", encoding="utf-8")
    uv = shutil.which("uv")
    assert uv, "seeded workspace regression requires real uv"

    def environment(name):
        return PythonEnvironment(
            uv=Path(uv), python=Path(sys.executable), destination=tmp_path / name,
            cache=tmp_path / f"{name}-cache", env=dict(os.environ), offline=True,
        )

    initial = tmp_path / "initial"
    lock_and_sync([retained], [], root=initial, source=source, seed_lock=None,
                  environment=environment("initial-env"))
    seed = initial / "uv.lock"
    before = seed.read_bytes()
    extended = tmp_path / "extended"
    candidate = environment("extended-env")
    lock_and_sync([earlier, retained], [], root=extended, source=source, seed_lock=seed,
                  environment=candidate)
    expected = f"plugin-sources/{_member_key(retained)}"
    assert tomllib.loads((extended / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"] == [expected]
    assert seed.read_bytes() == before
    probe = subprocess.run(
        [str(candidate.executable), "-I", "-c", "import replay_plugin; print(replay_plugin.VALUE)"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True,
    )
    assert probe.stdout.strip() == "recorded plugin bytes"


@pytest.mark.parametrize("prefer_seed", [False, True])
def test_divergent_buildable_members_across_profiles_fail_closed(tmp_path, prefer_seed):
    from pm.workspace import _generate_pyproject, _member_key

    source = tmp_path / "core"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "core"\nversion = "1"\n', encoding="utf-8"
    )
    plugins = []
    for home, value in (("a", 1), ("b", 2)):
        plugin = tmp_path / home / "plugins" / "same"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\nname = "same-plugin"\nversion = "1"\n'
            '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
            encoding="utf-8",
        )
        (plugin / "backend.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
        plugins.append(plugin)

    seed = None
    if prefer_seed:
        seed = tmp_path / "uv.lock"
        seed.write_text(
            'version = 1\n[[package]]\nname = "same-plugin"\nversion = "1"\n'
            f'source = {{ editable = "plugin-sources/{_member_key(plugins[1])}" }}\n',
            encoding="utf-8",
        )
    with pytest.raises(InstallError, match="differs across profiles"):
        _generate_pyproject(plugins, tmp_path / "generation", source=source, seed_lock=seed)


def test_virtual_member_without_project_metadata_gets_a_version(tmp_path):
    """A metadata-only member whose pyproject declares no [project] (lint config
    only) is emitted as ``[project] name = …``; uv rejects a [project] table
    without ``version`` (or ``dynamic``) at ``uv lock`` time, aborting the whole
    update. Generated virtual members carry the same inert version plugin-deps
    use."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "hermes-lcm"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text('[tool.ruff]\ntarget-version = "py311"\n', encoding="utf-8")
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == "0.0.0"


def test_virtual_member_declaring_dynamic_version_is_left_alone(tmp_path):
    """PEP 621 forbids supplying ``project.version`` while it is listed under
    ``project.dynamic`` ("You cannot provide a value for project.version and
    list it under project.dynamic at the same time"). A metadata-only member
    may legitimately declare ``dynamic = ["version"]``; the generated member
    must not gain a static version in that case, or ``uv lock`` rejects the
    conflicted manifest."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "dynver"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "dynver"\ndynamic = ["version"]\n', encoding="utf-8"
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "version" not in project
    assert project["dynamic"] == ["version"]


def test_buildable_member_without_a_version_gets_one_and_keeps_its_name(tmp_path):
    """uv rejects a [project] table with no version at parse time, buildable or not.
    A buildable member still keeps its declared name (uv checks it against its backend)."""
    import tomllib
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "noversion"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "noversion-plugin"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()
    member = _workspace_member(plugin, root, identity=plugin)
    project = tomllib.loads((member / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project == {"name": "noversion-plugin", "version": "0.0.0"}

"""Installed directory identity is shared by CLI, loader and dependency selection."""
import json

import pytest

from hermes_cli import plugins_cmd
from hermes_cli.plugins_discovery import collect_directory_manifests, gate_manifest
from hermes_cli.agent_plugins import PLUGIN_SCHEMA_V1
from pm.workspace import selected_plugin_dir


@pytest.fixture
def installed(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    (home / 'plugins').mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(plugins_cmd, '_toggle_plugin_toolset', lambda *a, **kw: None)
    monkeypatch.setattr(plugins_cmd, '_enable_consent', lambda *a, **kw: None)
    return home


@pytest.mark.parametrize('suffix', ['yaml', 'yml', 'json'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('symlink', [False, True])
def test_installed_key_matches_loader_and_pm(installed, tmp_path, suffix, nested, symlink):
    key = 'custom/probe' if nested else 'probe'
    source = tmp_path / 'source' if symlink else installed / 'plugins' / key
    source.mkdir(parents=True)
    if suffix == 'json':
        (source / 'plugin.json').write_text(json.dumps({'$schema': PLUGIN_SCHEMA_V1, 'name': 'probe-manifest', 'version': '1.0.0'}))
    else:
        (source / f'plugin.{suffix}').write_text('name: probe-manifest\nversion: 1.0.0\n')
    (source / '__init__.py').write_text('def register(ctx): pass\n')
    target = installed / 'plugins' / key
    if symlink:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    row = next(e for e in plugins_cmd._discover_all_plugins() if e[0] == 'probe-manifest')
    manifest = next(m for m in collect_directory_manifests() if m.name == 'probe-manifest')
    assert row[5] == manifest.key == key
    if suffix == 'json' and not nested:
        from hermes_cli.plugins_manifest import _portable_data_namespace
        assert manifest.data_namespace == _portable_data_namespace('probe-manifest')
    assert plugins_cmd._resolve_plugin_key(key) == key
    assert plugins_cmd._resolve_plugin_key('probe-manifest') == key
    assert selected_plugin_dir(installed / 'plugins', key) == target
    assert gate_manifest(manifest, set(), {key}).action == 'load'


@pytest.mark.parametrize('enable', [True, False])
def test_activation_canonicalizes_both_lists_and_is_idempotent(installed, monkeypatch, enable):
    p = installed / 'plugins' / 'probe'
    p.mkdir()
    (p / 'plugin.yaml').write_text('name: probe-manifest\nversion: 1.0.0\n')
    enabled, disabled = {'probe', 'probe-manifest'}, {'probe-manifest'}
    calls = []
    monkeypatch.setattr(plugins_cmd, '_get_enabled_set', lambda: enabled)
    monkeypatch.setattr(plugins_cmd, '_get_disabled_set', lambda: disabled)
    monkeypatch.setattr(plugins_cmd, '_disabled_by_update', lambda *a: False)

    def persist(key, *, enable, aliases, **kw):
        plugins_cmd._apply_activation(enabled, disabled, key, aliases, enable=enable)
        calls.append(key)

    monkeypatch.setattr(plugins_cmd, '_set_plugin_enabled', persist)
    assert plugins_cmd._activate_key('probe', enable=enable)
    assert (enabled, disabled) == ({'probe'}, set()) if enable else (enabled, disabled) == (set(), {'probe'})
    assert not plugins_cmd._activate_key('probe', enable=enable)
    assert calls == ['probe']


def test_canonical_directory_wins_over_another_manifest_alias(installed):
    for directory, name in [('probe', 'own-name'), ('aaa', 'probe')]:
        p = installed / 'plugins' / directory
        p.mkdir()
        (p / 'plugin.yaml').write_text(f'name: {name}\nversion: 1.0.0\n')
    assert plugins_cmd._find_plugin_entry('probe')[4] == installed / 'plugins' / 'probe'


def test_cli_repairs_mixed_selection_without_another_alias(installed, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from plugins.memory import _explicitly_disabled

    p = installed / 'plugins' / 'probe'
    p.mkdir()
    (p / 'plugin.yaml').write_text('name: probe-manifest\nversion: 1.0.0\n')
    cfg = load_config()
    cfg['plugins']['enabled'] = ['probe-manifest']
    cfg['plugins']['disabled'] = ['probe']
    save_config(cfg)

    def admit(enabled, disabled, **kw):
        cfg = load_config()
        cfg['plugins']['enabled'] = sorted(enabled)
        cfg['plugins']['disabled'] = sorted(disabled)
        save_config(cfg)

    monkeypatch.setattr('hermes_cli.plugins_admission.admit_plugin_set_change', admit)
    assert _explicitly_disabled('probe', p)
    plugins_cmd.cmd_enable('probe-manifest', allow_tool_override=False)
    cfg = load_config()['plugins']
    assert cfg['enabled'] == ['probe'] and cfg['disabled'] == []
    assert not _explicitly_disabled('probe', p)
    plugins_cmd.cmd_disable('probe')
    cfg = load_config()['plugins']
    assert cfg['enabled'] == [] and cfg['disabled'] == ['probe']


def test_duplicate_manifest_alias_is_not_guessed(installed):
    for directory in ['first', 'second']:
        p = installed / 'plugins' / directory
        p.mkdir()
        (p / 'plugin.yaml').write_text('name: same-manifest\nversion: 1.0.0\n')
    assert plugins_cmd._find_plugin_entry('same-manifest') is None

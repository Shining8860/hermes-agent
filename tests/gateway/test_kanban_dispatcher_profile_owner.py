"""The shared gateway dispatches under one enabled served profile, not its launch home."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


@pytest.fixture
def profile_fleet(tmp_path, monkeypatch):
    from agent import secret_scope

    home = tmp_path / '.hermes'
    homes = {'default': home}
    homes.update({name: home / 'profiles' / name for name in ('silas', 'mira', 'outside')})
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_PROFILE', 'default')
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.delenv('HERMES_KANBAN_DISPATCH_IN_GATEWAY', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    for name, path in homes.items():
        path.mkdir(parents=True, exist_ok=True)
        (path / '.env').write_text('HERMES_DISPATCH_TEST=' + name + '\n')
        (path / 'config.yaml').write_text(json.dumps({'terminal': {'timeout': 123 if name == 'silas' else 321}, 'kanban': {
            'dispatch_in_gateway': name in ('silas', 'outside'),
            'dispatch_interval_seconds': 7 if name == 'silas' else 91,
            'failure_limit': 4 if name == 'silas' else 2,
            'auto_decompose': False,
        }}))
    old_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        yield homes
    finally:
        secret_scope.set_multiplex_active(old_multiplex)


def _runner(homes, multiplex=True):
    runner = GatewayKanbanWatchersMixin()
    runner.config = SimpleNamespace(
        multiplex_profiles=multiplex,
        _runtime_profile_homes=[(name, path) for name, path in homes.items() if name != 'outside'],
    )
    return runner


def _set_enabled(homes, name, enabled):
    path = homes[name] / 'config.yaml'
    data = json.loads(path.read_text())
    data['kanban']['dispatch_in_gateway'] = enabled
    path.write_text(json.dumps(data))


def test_boot_uses_only_enabled_served_profile_and_binds_reload(profile_fleet):
    """Default=false must not mask Silas=true; an excluded true profile is irrelevant."""
    from hermes_constants import get_hermes_home

    runner = _runner(profile_fleet)
    try:
        boot = runner._kanban_dispatcher_boot()
        assert boot is not None
        load_config, kb, kanban_config = boot
        assert kanban_config['dispatch_interval_seconds'] == 7
        assert runner._owns_kanban_dispatcher_lock()
        assert kb.kanban_home() == profile_fleet['default']
        # The callback must enter its owner scope even when called outside the watcher.
        assert load_config()['kanban']['failure_limit'] == 4
        assert get_hermes_home() == profile_fleet['default']
    finally:
        runner._release_kanban_dispatcher_lock()


def test_all_served_dispatchers_disabled_does_not_open_lock(profile_fleet):
    _set_enabled(profile_fleet, 'silas', False)
    runner = _runner(profile_fleet)
    assert runner._kanban_dispatcher_boot() is None
    assert not (profile_fleet['default'] / 'kanban/.dispatcher.lock').exists()


@pytest.mark.parametrize('other', ['default', 'mira'])
def test_multiple_enabled_owners_fail_closed(profile_fleet, other):
    _set_enabled(profile_fleet, other, True)
    runner = _runner(profile_fleet)
    try:
        assert runner._kanban_dispatcher_boot() is None
        assert not runner._owns_kanban_dispatcher_lock()
    finally:
        runner._release_kanban_dispatcher_lock()


def test_standalone_does_not_adopt_other_profile_dispatch(profile_fleet):
    runner = _runner(profile_fleet, multiplex=False)
    assert runner._kanban_dispatcher_boot() is None


def test_standalone_keeps_its_own_enabled_profile(profile_fleet, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(profile_fleet['silas']))
    monkeypatch.setenv('HERMES_PROFILE', 'silas')
    runner = _runner(profile_fleet, multiplex=False)
    try:
        boot = runner._kanban_dispatcher_boot()
        assert boot is not None
        assert boot[2]['dispatch_interval_seconds'] == 7
        assert runner._kanban_dispatcher_profile.name == 'silas'
    finally:
        runner._release_kanban_dispatcher_lock()


def test_global_disable_and_contended_singleton_still_win(profile_fleet, monkeypatch):
    first = _runner(profile_fleet)
    second = _runner(profile_fleet)
    try:
        assert first._kanban_dispatcher_boot() is not None
        assert second._kanban_dispatcher_boot() is None
        assert not second._owns_kanban_dispatcher_lock()
        first._release_kanban_dispatcher_lock()
        monkeypatch.setenv('HERMES_KANBAN_DISPATCH_IN_GATEWAY', 'false')
        assert second._kanban_dispatcher_boot() is None
    finally:
        first._release_kanban_dispatcher_lock()
        second._release_kanban_dispatcher_lock()


def test_owner_thread_drops_request_context_but_keeps_owner_scope(profile_fleet):
    import asyncio
    from agent.delegation_context import delegated_child_context, is_delegated_child_context
    from agent.secret_scope import get_secret, current_secret_scope
    from gateway.kanban_watchers_common import _to_thread_process_service
    from hermes_constants import get_hermes_home
    from hermes_cli import kanban_db as kb
    from tools.terminal_scope import terminal_env

    runner = _runner(profile_fleet)
    try:
        assert runner._kanban_dispatcher_boot() is not None
        owner = runner._kanban_dispatcher_profile

        def inspect_owner():
            kb._assert_not_delegated_child_mutation()
            return (get_hermes_home(), get_secret('HERMES_DISPATCH_TEST'),
                    terminal_env('TERMINAL_TIMEOUT'), is_delegated_child_context(),
                    kb._CURRENT_BOARD_OVERRIDE.get())

        async def probe():
            token = kb._CURRENT_BOARD_OVERRIDE.set('request-only-board')
            try:
                with delegated_child_context():
                    observed = await _to_thread_process_service(owner.run, inspect_owner)
                    assert is_delegated_child_context()
                    assert kb._CURRENT_BOARD_OVERRIDE.get() == 'request-only-board'
                    return observed
            finally:
                kb._CURRENT_BOARD_OVERRIDE.reset(token)

        assert asyncio.run(probe()) == (profile_fleet['silas'], 'silas', '123', False, None)
        assert get_hermes_home() == profile_fleet['default']
        assert current_secret_scope() is None
    finally:
        runner._release_kanban_dispatcher_lock()


def test_owner_live_auto_decompose_toggle_is_not_launch_config(profile_fleet):
    from gateway.kanban_watchers_common import _resolve_auto_decompose_settings

    runner = _runner(profile_fleet)
    try:
        load_config, _, _ = runner._kanban_dispatcher_boot()
        path = profile_fleet['default'] / 'config.yaml'
        data = json.loads(path.read_text())
        data['kanban']['auto_decompose'] = True
        path.write_text(json.dumps(data))
        assert _resolve_auto_decompose_settings(load_config)[0] is False
        path = profile_fleet['silas'] / 'config.yaml'
        data = json.loads(path.read_text())
        data['kanban'].update(auto_decompose=True, auto_decompose_per_tick=1)
        path.write_text(json.dumps(data))
        assert _resolve_auto_decompose_settings(load_config) == (True, 1)
        path.write_text('kanban: [broken')
        assert _resolve_auto_decompose_settings(load_config)[0] is False
    finally:
        runner._release_kanban_dispatcher_lock()


def test_unreadable_served_config_does_not_activate_fallback_defaults(profile_fleet):
    (profile_fleet['mira'] / 'config.yaml').write_text('kanban: [broken')
    runner = _runner(profile_fleet)
    assert runner._kanban_dispatcher_boot() is None
    assert not runner._owns_kanban_dispatcher_lock()


def test_cancellation_during_initial_delay_releases_lock(profile_fleet, monkeypatch):
    import asyncio
    from gateway import kanban_watchers

    runner = _runner(profile_fleet)
    runner._running = True

    async def cancel_delay(delay):
        raise asyncio.CancelledError()

    monkeypatch.setattr(kanban_watchers.asyncio, 'sleep', cancel_delay)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._kanban_dispatcher_watcher())
    assert not runner._owns_kanban_dispatcher_lock()
    second = _runner(profile_fleet)
    try:
        assert second._kanban_dispatcher_boot() is not None
    finally:
        second._release_kanban_dispatcher_lock()


def test_real_empty_board_ticks_write_only_in_selected_owner_scope(profile_fleet, monkeypatch):
    import asyncio
    from agent.secret_scope import get_secret
    from gateway import kanban_watchers
    from hermes_cli import kanban_db as kb, kanban_db_dispatch as kbd
    from hermes_constants import get_hermes_home

    kb.init_db(board='default')
    real_dispatch_once = kbd.dispatch_once
    seen = []
    runner = _runner(profile_fleet)
    runner._running = True

    def record_tick(conn, **kwargs):
        seen.append((get_hermes_home(), get_secret('HERMES_DISPATCH_TEST')))
        return real_dispatch_once(conn, **kwargs)

    async def no_delay(delay):
        return None

    tick_intervals = []

    async def stop_after_two(interval):
        tick_intervals.append(interval)
        if len(tick_intervals) >= 2:
            runner._running = False

    monkeypatch.setattr(kanban_watchers.asyncio, 'sleep', no_delay)
    monkeypatch.setattr(runner, '_sleep_between_ticks', stop_after_two)
    monkeypatch.setattr(kbd, 'reap_worker_zombies', list)
    monkeypatch.setattr(kbd, 'dispatch_once', record_tick)
    asyncio.run(runner._kanban_dispatcher_watcher())
    assert seen == [(profile_fleet['silas'], 'silas')] * 2
    assert not runner._owns_kanban_dispatcher_lock()

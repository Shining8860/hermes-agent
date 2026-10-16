"""Select and bind the sole profile owning a process-global Kanban dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gateway.kanban_watchers_common import logger


@dataclass(frozen=True)
class DispatcherProfile:
    """A fixed owner; threads enter this scope AFTER dropping request context."""

    name: str
    home: Path
    config: dict

    def run(self, func: Callable[..., Any], /, *args: Any) -> Any:
        from gateway.run import _profile_runtime_scope

        with _profile_runtime_scope(self.home):
            return func(*args)

    def load_config(self) -> dict:
        from hermes_cli.config import FailedConfigRead, load_config

        config = self.run(load_config)
        if isinstance(config, FailedConfigRead) or not isinstance(config, dict):
            raise ValueError('dispatcher profile configuration is unreadable')
        return config


def select_dispatcher_profile(runner: Any) -> DispatcherProfile | None:
    """An enabled served profile owns the singleton, regardless of the launcher.

    Multiple enabled profiles are ambiguous: fail closed before opening a lock
    or touching a board. Standalone gateways never adopt another profile's flag.
    Settings and live auto-decompose reads stay with the selected owner.
    """
    from gateway.run import _multiplex_profile_homes
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_process_hermes_home

    config = getattr(runner, 'config', None)
    if getattr(config, 'multiplex_profiles', False):
        homes = _multiplex_profile_homes(config)
    else:
        # launch profile, pre-identity: standalone dispatch cannot adopt a routed home.
        homes = [(get_active_profile_name() or 'default', get_process_hermes_home())]
    enabled = []
    for name, home in homes:
        owner = DispatcherProfile(name, Path(home), {})
        try:
            cfg = owner.load_config()
            kanban = cfg.get('kanban', {}) if isinstance(cfg, dict) else {}
            if not isinstance(kanban, dict):
                raise TypeError('kanban settings must be a mapping')
        except Exception as exc:  # health: allow BLE001 -- fail closed; parser tracebacks can expose YAML credentials
            # Parser exceptions can contain YAML source, including credentials.
            logger.warning('kanban dispatcher: config unavailable for profile %s (%s); disabled',
                           name, type(exc).__name__)
            return None
        if kanban.get('dispatch_in_gateway', True):
            enabled.append(DispatcherProfile(name, Path(home), cfg))
    if len(enabled) > 1:
        logger.error('kanban dispatcher: multiple served profiles enable dispatch (%s); '
                     'enable kanban.dispatch_in_gateway for exactly one profile; disabled',
                     ', '.join(owner.name for owner in enabled))
        return None
    if not enabled:
        logger.info('kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false '
                    'for all served profiles')
        return None
    return enabled[0]

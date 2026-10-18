"""The gateway owns toolset validation; CLI startup must not load plugins (PLG-7).

The old overlay's discover-before-warning hook is superseded. Gateway-side
unknown-name refusal is covered by test_cli_profile_toolsets.py.
"""
from __future__ import annotations

import pytest

import cli as cli_module
from hermes_cli import plugins as plugins_module


@pytest.mark.parametrize("name", ["plugin_demo", "definitely_unknown"])
def test_init_defers_toolset_validation_to_gateway(monkeypatch, name):
    rendered: list[str] = []

    def no_discovery(*_args, **_kwargs):
        pytest.fail("CLI toolset initialization must not discover plugins")

    for entrypoint in ("discover_plugins", "get_plugin_manager", "get_plugin_toolset_keys_nowait"):
        monkeypatch.setattr(plugins_module, entrypoint, no_discovery)
    monkeypatch.setitem(cli_module.CLI_CONFIG, "agent", {"disabled_toolsets": []})

    session = object.__new__(cli_module.HermesCLI)
    session._console_print = rendered.append
    session._init_toolsets([name])

    assert session.enabled_toolsets == [name]
    assert session._explicit_toolsets is True
    assert rendered == []

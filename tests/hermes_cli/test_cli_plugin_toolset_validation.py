from __future__ import annotations

import cli as cli_module
from hermes_cli import plugins as plugins_module


def test_init_toolsets_discovers_plugins_before_warning(monkeypatch):
    discovered = False
    rendered: list[str] = []

    def fake_discover_plugins() -> None:
        nonlocal discovered
        discovered = True

    def fake_validate_toolset(name: str) -> bool:
        return name == "plugin_demo" and discovered

    monkeypatch.setattr(plugins_module, "discover_plugins", fake_discover_plugins)
    monkeypatch.setattr(cli_module, "validate_toolset", fake_validate_toolset)
    monkeypatch.setitem(cli_module.CLI_CONFIG, "agent", {"disabled_toolsets": []})
    monkeypatch.setitem(cli_module.CLI_CONFIG, "mcp_servers", {})

    session = object.__new__(cli_module.HermesCLI)
    session._console_print = rendered.append

    session._init_toolsets(["plugin_demo"])

    assert discovered is True
    assert session.enabled_toolsets == ["plugin_demo"]
    assert rendered == []


def test_init_toolsets_still_warns_for_unknown_name_after_discovery(monkeypatch):
    rendered: list[str] = []

    monkeypatch.setattr(plugins_module, "discover_plugins", lambda: None)
    monkeypatch.setattr(cli_module, "validate_toolset", lambda _name: False)
    monkeypatch.setitem(cli_module.CLI_CONFIG, "agent", {"disabled_toolsets": []})
    monkeypatch.setitem(cli_module.CLI_CONFIG, "mcp_servers", {})

    session = object.__new__(cli_module.HermesCLI)
    session._console_print = rendered.append

    session._init_toolsets(["definitely_unknown"])

    assert rendered == ["[bold red]Warning: Unknown toolsets: definitely_unknown[/]"]

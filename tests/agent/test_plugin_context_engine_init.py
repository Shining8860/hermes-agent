"""Tests that plugin context engines get update_model() called during init.

Regression test for #9071 — plugin engines were never initialized with
context_length, causing the CLI status bar to show 'ctx --'.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_engine import ContextEngine


class _StubEngine(ContextEngine):
    """Minimal concrete context engine for testing."""

    @property
    def name(self) -> str:
        return "stub"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None):
        return messages


class _ToolEngine(_StubEngine):
    def get_tool_schemas(self):
        return [
            {
                "name": "stub_recover",
                "description": "Recover context from the stub engine.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]


def test_plugin_engine_gets_context_length_on_init():
    """Plugin context engine should have context_length set during AIAgent init."""
    engine = _StubEngine()
    assert engine.context_length == 0  # ABC default before fix

    cfg = {"context": {"engine": "stub"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.context_compressor is engine
    assert engine.context_length == 204_800
    assert engine.threshold_tokens == int(204_800 * engine.threshold_percent)


def test_active_context_engine_tools_survive_explicit_platform_toolsets():
    """LCM-style recovery tools must survive saved `hermes tools` lists."""
    engine = _ToolEngine()
    cfg = {
        "context": {"engine": "stub"},
        "platform_toolsets": {"cli": ["web", "terminal"]},
        "agent": {},
    }

    from hermes_cli.tools_config import _get_platform_tools

    enabled_toolsets = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    assert "context_engine" in enabled_toolsets

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=sorted(enabled_toolsets),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert "stub_recover" in getattr(agent, "valid_tool_names", set())
    assert "stub_recover" in {
        tool.get("function", {}).get("name")
        for tool in getattr(agent, "tools", [])
    }


class _DeferredToolEngine(_ToolEngine):
    def __init__(self):
        self.calls = []
        self.extra_schemas = []

    def get_tool_schemas(self):
        return super().get_tool_schemas() + [{
            "name": "lcm_inspect",
            "description": "Inspect context engine state.",
            "parameters": {"type": "object", "properties": {"section": {"type": "string"}},
                           "required": ["section"]},
        }] + self.extra_schemas

    def handle_tool_call(self, name, args, *, messages=None, **kwargs):
        self.calls.append((name, args, messages))
        return json.dumps({"engine": self.name, "tool": name})


@pytest.mark.parametrize("refresh", [False, True])
def test_deferred_lcm_tool_is_catalogued_and_executable(refresh):
    """An engine-only schema uses the bridge without losing session-local dispatch."""
    engine = _DeferredToolEngine()
    cfg = {
        "context": {"engine": "stub"}, "agent": {},
        "tools": {"tool_search": {"defer": ["lcm_inspect", "lcm_doctor", "registry_lookup"]}},
    }
    registry_defs = [{"type": "function", "function": {
        "name": "registry_lookup", "description": "Existing deferred registry lookup.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", side_effect=lambda **kw:
              registry_defs if kw.get("skip_tool_search_assembly") and kw.get("enabled_toolsets") != [] else []),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent
        from tools.mcp_tool_agent import refresh_agent_mcp_tools

        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=["context_engine"], quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
        assert "lcm_inspect" not in {t["function"]["name"] for t in agent.tools}
        if refresh:
            refresh_agent_mcp_tools(agent)
            prefix = list(agent.tools)
            engine.extra_schemas.append({"name": "lcm_doctor", "description": "Diagnose engine state.",
                                         "parameters": {"type": "object", "properties": {}}})
            refresh_agent_mcp_tools(agent, preserve_prefix=True)
            assert agent.tools == prefix
            assert "lcm_doctor" in agent._context_engine_tool_names
        names = [t["function"]["name"] for t in agent.tools]
        assert "lcm_inspect" not in names
        assert "lcm_inspect" not in agent.valid_tool_names
        assert "stub_recover" in names
        assert {"tool_search", "tool_describe", "tool_call"} <= set(names)
        assert len(names) == len(set(names))

        def execute(name, args):
            messages = []
            tc = SimpleNamespace(id=f"test-{name}", function=SimpleNamespace(
                name=name, arguments=json.dumps(args)))
            agent._execute_tool_calls(SimpleNamespace(tool_calls=[tc]), messages, "test")
            return messages[-1]["content"]

        assert "registry_lookup" not in names
        manifest = next(t["function"]["description"] for t in agent.tools
                        if t["function"]["name"] == "tool_search")
        assert "lcm_inspect" in manifest and "registry_lookup" in manifest
        assert "lcm_inspect" in execute("tool_search", {"queries": ["lcm_inspect"]})
        assert "registry_lookup" in execute("tool_search", {"queries": ["registry_lookup"]})
        assert "Inspect context engine state." in execute("tool_describe", {"names": ["lcm_inspect"]})
        invalid = execute("tool_call", {"calls": [{"name": "lcm_inspect", "arguments": {}}]})
        assert "missing required argument" in invalid
        assert engine.calls == []
        result = execute("tool_call", {"calls": [{"name": "lcm_inspect", "arguments": {"section": "tools"}}]})
        assert '"engine": "stub"' in result
        assert engine.calls[0][:2] == ("lcm_inspect", {"section": "tools"})
        assert engine.calls[0][2] is not None

        # A restricted sibling cannot discover or invoke the first agent's engine tools.
        restricted = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        from agent.tool_executor import _unwrap_tool_search_call
        _, _, blocked = _unwrap_tool_search_call(restricted, "tool_call", {
            "calls": [{"name": "lcm_inspect", "arguments": {}}],
        })
        assert "not available in this session" in blocked
        assert len(engine.calls) == 1


@pytest.mark.parametrize("settings", [
    {"defer": []}, {"enabled": "off", "defer": ["lcm_inspect"]},
])
@pytest.mark.parametrize("registry_collision", [False, True])
def test_engine_disclosure_preserves_eager_mode_and_registry_ownership(settings, registry_collision):
    from agent.context_engine_tools import inject_engine_tools

    engine = _DeferredToolEngine()
    registry_defs = [{"type": "function", "function": {
        "name": "lcm_inspect", "description": "Registry owns this tool.", "parameters": {},
    }}] if registry_collision else []
    agent = SimpleNamespace(context_compressor=engine, enabled_toolsets=["context_engine"])
    tools, names = [], set()
    cfg = {"tools": {"tool_search": settings}}
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("model_tools.get_tool_definitions", return_value=registry_defs),
    ):
        owned = inject_engine_tools(agent, tools, names)

    assert "lcm_inspect" in names
    assert "stub_recover" in owned
    assert ("lcm_inspect" in owned) is not registry_collision
    assert not {"tool_search", "tool_describe", "tool_call"} & names
    if registry_collision:
        assert next(t["function"]["description"] for t in tools
                    if t["function"]["name"] == "lcm_inspect") == "Registry owns this tool."


def _codex_agent_kwargs():
    return dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="test-key-1234567890",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_codex_gpt55_autoraise_suppressed_for_plugin_engine():
    """Codex gpt-5.5 autoraise must not fire when an external engine is active.

    Regression test for #44439 — the host compression threshold (including
    the 0.85 autoraise) never reaches a plugin context engine, so the notice
    announced a change that did not apply.
    """
    engine = _StubEngine()
    cfg = {
        "context": {"engine": "stub"},
        "compression": {"enabled": True, "threshold": 0.75},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.context_engine.load_context_engine", return_value=engine),
        patch("agent.model_metadata.get_model_context_length", return_value=272_000),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent.context_compressor is engine
    assert agent._compression_threshold_autoraised is None
    assert agent._compression_warning is None
    # The engine's own policy is untouched by the host threshold.
    assert engine.threshold_percent == 0.75


def test_codex_gpt55_autoraise_still_applies_to_builtin_compressor():
    """Stock built-in compressor keeps the 50% → 85% Codex gpt-5.5 autoraise."""
    cfg = {
        "compression": {"enabled": True, "threshold": 0.50},
        "agent": {},
    }

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.context_compressor.get_model_context_length", return_value=272_000),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(**_codex_agent_kwargs())

    assert agent._compression_threshold_autoraised == {"model": "gpt-5.5", "from": 0.50, "to": 0.85}
    assert agent.context_compressor.threshold_percent == 0.85
    # Gateway parity: the notice is stashed for replay on turn 1.
    assert agent._compression_warning

"""Execute the overlay catalog path, not just its lazy-importing module."""
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest


def schema(name):
    return {"type": "function", "function": {
        "name": name, "description": name,
        "parameters": {"type": "object", "properties": {
            "tasks": {"type": "array"}, "action": {"type": "string"},
            "subagent_id": {"type": "string"}, "message": {"type": "string"},
        }},
    }}


@pytest.mark.parametrize("oneshot", [False, True])
def test_catalog_and_session_surface_keep_variants_and_session_drops(monkeypatch, oneshot):
    module = import_module("agent.context_engine_tools")
    agent = SimpleNamespace(
        platform="cli", side_agent=True, _oneshot_session=oneshot,
        enabled_toolsets=["context_engine"], disabled_toolsets=[],
        context_compressor=SimpleNamespace(get_tool_schemas=lambda: [schema("engine_lookup")]),
        _context_engine_tool_names={"engine_lookup"},
    )
    definitions = [schema(n) for n in ("delegate_task", "skill_manage", "manage_connections")]
    original = deepcopy(definitions)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: definitions)
    session = module._session_tool_defs(agent, iter(definitions))
    catalog = module.catalog_tool_defs(agent)
    assert [td["function"]["name"] for td in session] == ["delegate_task", "skill_manage"]
    assert [td["function"]["name"] for td in catalog] == ["delegate_task", "skill_manage", "engine_lookup"]
    props = session[0]["function"]["parameters"]["properties"]
    assert ("action" not in props) is oneshot
    assert "tasks" in props
    assert module._session_tool_defs(agent, session) == session
    assert module._session_tool_defs(agent, catalog) == catalog
    assert definitions == original


def test_deferred_scope_refreshes_registry_without_losing_owned_engine_tools(monkeypatch):
    from agent.tool_executor import _tool_search_scoped_names
    from tools.tool_search import ToolSearchConfig

    agent = SimpleNamespace(
        enabled_toolsets=["context_engine", "mcp-probe"], disabled_toolsets=[],
        context_compressor=SimpleNamespace(get_tool_schemas=lambda: [schema("engine_lookup")]),
        _context_engine_tool_names={"engine_lookup"},
    )
    monkeypatch.setattr("tools.tool_search.load_config_readonly", lambda: ToolSearchConfig.from_raw({
        "defer": ["engine_lookup", "registry_lookup"],
    }))
    definitions = []
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: list(definitions))
    assert _tool_search_scoped_names(agent) == {"engine_lookup"}
    definitions.append(schema("registry_lookup"))
    assert _tool_search_scoped_names(agent) == {"engine_lookup", "registry_lookup"}
    definitions.clear()
    assert _tool_search_scoped_names(agent) == {"engine_lookup"}

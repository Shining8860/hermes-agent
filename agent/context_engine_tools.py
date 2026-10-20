"""Session-local context-engine schemas in progressive tool disclosure.

Engine handlers own live agent state, so their schemas must not enter the process-wide
registry just to make them searchable. Assemble and query the same scoped catalog instead.
"""

import logging

from agent.memory_manager import normalize_tool_schema

logger = logging.getLogger(__name__)


def engine_tool_defs(agent) -> list:
    """Normalized engine schemas, subject to the same toolset gate as injection."""
    enabled = getattr(agent, "enabled_toolsets", None)
    engine = getattr(agent, "context_compressor", None)
    if engine is None or (enabled is not None and "context_engine" not in enabled):
        return []
    getter = getattr(engine, "get_tool_schemas", None)
    if not callable(getter):
        return []
    definitions = []
    for raw in getter():
        schema = normalize_tool_schema(raw)
        if schema is None:
            logger.warning("Context engine returned a tool schema with no resolvable name; skipping (%r)", raw)
        else:
            definitions.append({"type": "function", "function": schema})
    return definitions


def catalog_tool_defs(agent) -> list:
    """Registry catalog plus only the engine names owned by THIS agent."""
    from model_tools import get_tool_definitions

    definitions = list(get_tool_definitions(
        enabled_toolsets=getattr(agent, "enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        quiet_mode=True, skip_tool_search_assembly=True,
    ) or [])
    registered = {td["function"]["name"] for td in definitions}
    owned = getattr(agent, "_context_engine_tool_names", set())
    definitions.extend(td for td in engine_tool_defs(agent)
                       if td["function"]["name"] in owned and td["function"]["name"] not in registered)
    return definitions


def inject_engine_tools(agent, tools_list: list, name_set: set) -> set:
    """Stage engine routing names and reassemble the complete deferred catalog."""
    from model_tools import get_tool_definitions
    from tools.tool_search import assemble_tool_defs, load_config_readonly

    engine_defs = engine_tool_defs(agent)
    if not engine_defs:
        return set()
    # Include hidden registry schemas too: assembling the already-collapsed wire list
    # would replace its bridge manifest with an engine-only catalog.
    definitions = {td["function"]["name"]: td for td in get_tool_definitions(
        enabled_toolsets=getattr(agent, "enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        quiet_mode=True, skip_tool_search_assembly=True,
    ) or []}
    definitions.update((td["function"]["name"], td) for td in tools_list)
    owned = set()
    for td in engine_defs:
        name = td["function"]["name"]
        if name not in definitions:
            definitions[name] = td
            owned.add(name)
    tools_list[:] = assemble_tool_defs(
        list(definitions.values()),
        context_length=getattr(agent.context_compressor, "context_length", None),
        config=load_config_readonly(),
    ).tool_defs
    name_set.clear()
    name_set.update(td["function"]["name"] for td in tools_list)
    return owned


def dispatch_catalog_read(agent, args, ctx, *, name):
    """Shared sequential/concurrent inline bridge reads, with session identity intact."""
    from tools import tool_search
    from tools.managed_gateway_identity import CallRef

    dispatch = {"tool_search": tool_search.dispatch_tool_search,
                "tool_describe": tool_search.dispatch_tool_describe}[name]
    return dispatch(args, current_tool_defs=catalog_tool_defs(agent), call=CallRef(
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        tool_call_id=ctx.tool_call_id,
    ))

"""Per-channel ``disabled_toolsets`` (``platforms.<platform>.channel_overrides[<id>]``).

An operator who configures a toolset deny for one chat expects exactly that chat — and only that chat —
to lose the toolsets, on every path that builds an agent for it (interactive turn, background task,
hosted-room policy rebuild). The channel layer is CUMULATIVE over exact keys and the ``group:*``
wildcard and sits ON TOP of the global ``agent.disabled_toolsets``, which keeps its own effect.

The tests also pin the two failure modes this path exists to end: an unknown key inside a
``channel_overrides`` entry, and a deny name that resolves to no toolset, were both silent no-ops.
"""

import pytest

from gateway import config as gateway_config
from gateway.config import (
    ChannelOverride,
    Platform,
    PlatformConfig,
    channel_override_deny_toolsets,
    merge_deny_toolsets,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


CHAT_ID = "chat-a"
OTHER_CHAT_ID = "chat-b"
GROUP_ID = "group-1"
DM_ID = "dm-1"

# An explicit platform list keeps the enabled set small and stable; ``debugging`` is a composite whose
# tools cover ``file``/``terminal``/``web``, so a deny of it must remove those toolsets too.
BASE_CONFIG = {"platform_toolsets": {"telegram": ["web", "terminal", "file", "memory", "debugging"]}}


@pytest.fixture(autouse=True)
def _clear_warning_dedupe():
    """The override warnings are once-per-process; a fresh bucket per test keeps caplog meaningful."""
    gateway_config._WARNED_CHANNEL_OVERRIDES.clear()
    yield
    gateway_config._WARNED_CHANNEL_OVERRIDES.clear()


def _source(chat_id: str, chat_type: str = "group") -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type=chat_type)


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: None
    return runner


def _config(overrides: dict, *, global_deny=None) -> dict:
    config = {**BASE_CONFIG, "platforms": {"telegram": {"channel_overrides": overrides}}}
    if global_deny is not None:
        config["agent"] = {"disabled_toolsets": global_deny}
    return config


def _resolve(config: dict, source: SessionSource):
    return _runner()._resolve_turn_toolsets(config, source, "telegram")


class TestExactChannelDeny:
    """Acceptance 1: an exact key denies that chat and nothing else."""

    def test_exact_key_denies_only_that_chat(self):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}})
        enabled, disabled = _resolve(config, _source(CHAT_ID))
        assert "memory" in disabled
        assert "memory" not in enabled

        other_enabled, other_disabled = _resolve(config, _source(OTHER_CHAT_ID))
        assert other_disabled is None
        assert "memory" in other_enabled

    def test_thread_and_parent_keys_accumulate(self):
        """A thread inherits its parent's deny: both keys match, both apply."""
        config = _config({
            "parent-chan": {"disabled_toolsets": ["memory"]},
            "thread-77": {"disabled_toolsets": ["web"]},
        })
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id="thread-77", chat_type="thread",
            parent_chat_id="parent-chan",
        )
        enabled, disabled = _resolve(config, source)
        assert set(disabled) == {"memory", "web"}
        assert "web" not in enabled and "memory" not in enabled

    def test_string_and_list_normalisation(self):
        """YAML list, JSON-array string (what ``hermes config set`` writes) and one scalar name."""
        assert ChannelOverride.from_dict({"disabled_toolsets": ["memory", "web"]}).disabled_toolsets == [
            "memory", "web"]
        assert ChannelOverride.from_dict({"disabled_toolsets": "['memory', 'web']"}).disabled_toolsets == [
            "memory", "web"]
        assert ChannelOverride.from_dict({"disabled_toolsets": "memory"}).disabled_toolsets == ["memory"]
        assert ChannelOverride.from_dict({"disabled_toolsets": "memory, web"}).disabled_toolsets == [
            "memory, web"]


    def test_deny_reaches_the_tool_schemas_the_agent_sees(self):
        """End to end: the resolved pair removes the tool, not just the toolset name."""
        from model_tools import get_tool_definitions

        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}})
        enabled, disabled = _resolve(config, _source(CHAT_ID))
        names = {
            schema["function"]["name"]
            for schema in get_tool_definitions(
                enabled_toolsets=enabled, disabled_toolsets=disabled, quiet_mode=True)
        }
        assert "memory" not in names

        other_enabled, other_disabled = _resolve(config, _source(OTHER_CHAT_ID))
        other_names = {
            schema["function"]["name"]
            for schema in get_tool_definitions(
                enabled_toolsets=other_enabled, disabled_toolsets=other_disabled, quiet_mode=True)
        }
        assert "memory" in other_names


class TestGroupWildcard:
    """Acceptance 2: ``group:*`` reaches groups and leaves DMs alone."""

    def test_group_wildcard_hits_group(self):
        config = _config({"group:*": {"disabled_toolsets": ["memory"]}})
        enabled, disabled = _resolve(config, _source(GROUP_ID, chat_type="group"))
        assert disabled == ["memory"]
        assert "memory" not in enabled

    @pytest.mark.parametrize("chat_type", ["dm", "direct"])
    def test_group_wildcard_never_hits_a_dm(self, chat_type):
        config = _config({"group:*": {"disabled_toolsets": ["memory"]}})
        enabled, disabled = _resolve(config, _source(DM_ID, chat_type=chat_type))
        assert disabled is None
        assert "memory" in enabled

    def test_unknown_wildcard_is_reported_not_ignored(self, caplog):
        for wildcard in ("dm:*", "*"):
            with caplog.at_level("WARNING"):
                config = _config({wildcard: {"disabled_toolsets": ["memory"]}})
                assert _resolve(config, _source(DM_ID, chat_type="dm"))[1] is None
            assert any("unsupported channel wildcard" in record.getMessage() for record in caplog.records)

    def test_group_wildcard_reports_keys_it_cannot_apply(self, caplog):
        """A wildcard model/prompt would read as configured and never apply — say so."""
        config = _config({"group:*": {"model": "openai/gpt-5-mini", "disabled_toolsets": ["memory"]}})
        with caplog.at_level("WARNING"):
            _, disabled = _resolve(config, _source(GROUP_ID, chat_type="group"))
        assert disabled == ["memory"]
        assert any("only disabled_toolsets applies to the group wildcard" in record.getMessage()
                   for record in caplog.records)


class TestCumulativeAndGlobal:
    """Acceptance 3 + 4: exact and wildcard add up; the global list keeps its own effect."""

    def test_exact_and_wildcard_accumulate(self):
        config = _config({
            "group:*": {"disabled_toolsets": ["memory"]},
            GROUP_ID: {"disabled_toolsets": ["web"]},
        })
        enabled, disabled = _resolve(config, _source(GROUP_ID, chat_type="group"))
        assert set(disabled) == {"memory", "web"}
        assert "memory" not in enabled and "web" not in enabled

    def test_global_agent_disabled_still_applies(self):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}}, global_deny=["browser"])
        enabled, disabled = _resolve(config, _source(CHAT_ID))
        assert set(disabled) == {"browser", "memory"}
        assert "memory" not in enabled and "browser" not in enabled

    def test_channel_deny_does_not_leak_into_global_for_other_chats(self):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}}, global_deny=["browser"])
        enabled, disabled = _resolve(config, _source(OTHER_CHAT_ID))
        assert disabled == ["browser"]
        assert "memory" in enabled and "browser" not in enabled

    def test_composite_deny_prunes_the_toolsets_whose_tools_vanish(self):
        """Denying ``debugging`` must remove file/terminal/web from the enabled set, not just its name."""
        config = _config({CHAT_ID: {"disabled_toolsets": ["debugging"]}})
        enabled, disabled = _resolve(config, _source(CHAT_ID))
        assert "debugging" in disabled
        assert not {"debugging", "terminal", "web", "file"} & set(enabled)


class TestUnknownKeysAreLoud:
    """Acceptance 5: an unknown key inside an override is reported instead of silently dropped."""

    def test_unknown_key_warns(self, caplog):
        with caplog.at_level("WARNING"):
            ChannelOverride.from_dict({"disabled_toolset": ["memory"]}, where="channel_overrides[42]")
        messages = [record.getMessage() for record in caplog.records]
        assert any("unknown key(s) 'disabled_toolset' ignored" in message for message in messages)

        # The near-miss really does nothing, which is why the warning has to exist.
        assert ChannelOverride.from_dict({"disabled_toolset": ["memory"]}).disabled_toolsets == []

    def test_unknown_key_warns_through_platform_config(self, caplog):
        with caplog.at_level("WARNING"):
            parsed = PlatformConfig.from_dict({
                "enabled": True,
                "channel_overrides": {CHAT_ID: {"toolsets": ["memory"]}},
            })
        assert parsed.channel_overrides[CHAT_ID].disabled_toolsets == []
        assert any("unknown key(s) 'toolsets'" in record.getMessage() for record in caplog.records)

    def test_roundtrip_keeps_deny_and_drops_unknown_key(self):
        override = ChannelOverride.from_dict({
            "model": "openrouter/x",
            "disabled_toolsets": ["memory", "web"],
            "system_prompts": "typo",
        })
        roundtripped = override.to_dict()
        assert roundtripped == {"model": "openrouter/x", "disabled_toolsets": ["memory", "web"]}
        assert ChannelOverride.from_dict(roundtripped) == override

    def test_empty_deny_is_not_persisted(self):
        assert ChannelOverride(model="openrouter/x").to_dict() == {"model": "openrouter/x"}

    def test_non_mapping_override_warns(self, caplog):
        with caplog.at_level("WARNING"):
            parsed = PlatformConfig.from_dict({"channel_overrides": {CHAT_ID: ["memory"]}})
        assert parsed.channel_overrides == {}
        assert any("override must be a mapping" in record.getMessage() for record in caplog.records)


class TestOneResolutionSeam:
    """Acceptance 6: turn and background task resolve through the same behavior."""

    def test_interactive_display_settings_use_the_seam(self):
        """The interactive path's per-turn bundle carries exactly the seam's two lists."""
        from unittest.mock import patch

        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}}, global_deny=["browser"])
        runner = _runner()
        with patch("gateway.run._load_gateway_config", return_value=config):
            display = runner._run_agent_display_settings(_source(CHAT_ID))
        assert (display.enabled_toolsets, display.disabled_toolsets) == _resolve(
            config, _source(CHAT_ID))

    def test_same_config_same_set_every_call(self):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}}, global_deny=["browser"])
        assert _resolve(config, _source(CHAT_ID)) == _resolve(config, _source(CHAT_ID))

    def test_unknown_toolset_name_is_logged_and_never_grants(self, caplog):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memeory"]}})
        with caplog.at_level("ERROR"):
            enabled, disabled = _resolve(config, _source(CHAT_ID))
        errors = [record.getMessage() for record in caplog.records if record.levelname == "ERROR"]
        assert any("unknown toolset 'memeory'" in message for message in errors)
        # Fail-closed: the typo is never read as permission — the enabled set is unchanged, not widened.
        assert "memory" in enabled
        assert "memeory" in disabled

    def test_unknown_toolset_does_not_discard_the_resolvable_restriction(self, caplog):
        config = _config({CHAT_ID: {"disabled_toolsets": ["memeory", "memory"]}})
        with caplog.at_level("ERROR"):
            enabled, disabled = _resolve(config, _source(CHAT_ID))
        assert "memory" in disabled and "memory" not in enabled


class TestBackgroundTaskParity:
    """Acceptance 6, behavioural: the /bg path hands AIAgent the very set the seam resolved."""

    @pytest.mark.asyncio
    async def test_background_task_gets_the_same_toolset_set(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from gateway.hooks import HookRegistry

        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}}, global_deny=["browser"])
        expected = _resolve(config, _source(CHAT_ID))

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._voice_mode = {}
        runner._session_db = None
        runner._reasoning_config = None
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._running_agents = {}
        runner._background_tasks = set()
        runner.session_store = MagicMock()
        runner.session_store.get_model_override.return_value = None
        runner.hooks = HookRegistry()  # type: ignore[assignment]
        adapter = AsyncMock()
        adapter.toolsets_for_source = MagicMock(return_value=None)
        adapter.extract_media = MagicMock(return_value=([], "ok"))
        adapter.extract_images = MagicMock(return_value=([], "ok"))
        runner.adapters[Platform.TELEGRAM] = adapter

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value=config), \
             patch("run_agent.AIAgent") as MockAgent:
            instance = MagicMock()
            instance.run_conversation.return_value = {"final_response": "ok", "messages": []}
            MockAgent.return_value = instance
            await runner._run_background_task("hi", _source(CHAT_ID), "bg_test")

        kwargs = MockAgent.call_args.kwargs
        assert (kwargs["enabled_toolsets"], kwargs["disabled_toolsets"]) == expected


class TestNoMidTurnChange:
    """Acceptance 7: the running turn keeps its toolset snapshot; a config edit lands next turn."""

    def test_resolution_is_a_snapshot_per_turn(self):
        before = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}})
        enabled_before, disabled_before = _resolve(before, _source(CHAT_ID))
        assert disabled_before == ["memory"] and "memory" not in enabled_before
        frozen_enabled = list(enabled_before)

        # The operator edits config while the turn runs (same dict the next turn will re-read).
        before["platforms"]["telegram"]["channel_overrides"][CHAT_ID] = {
            "disabled_toolsets": ["memory", "debugging"]}

        # The running turn keeps the list it was built with: nothing re-reads config mid-turn.
        assert enabled_before == frozen_enabled
        assert disabled_before == ["memory"]

        # Next turn sees the new list, and the agent-cache signature changes with it (rebuild, not reuse).
        enabled_after, disabled_after = _resolve(before, _source(CHAT_ID))
        assert set(disabled_after) == {"memory", "debugging"}
        assert set(enabled_after) != set(enabled_before)

        signature = GatewayRunner._agent_config_signature.__func__ if hasattr(
            GatewayRunner._agent_config_signature, "__func__") else GatewayRunner._agent_config_signature
        sig_before = signature("m", {}, enabled_before, "p")
        sig_after = signature("m", {}, enabled_after, "p")
        assert sig_before != sig_after

    def test_disabled_list_passed_to_the_agent_is_the_resolved_object(self):
        """The agent gets the very list the seam returned — nothing re-reads config after construction."""
        config = _config({CHAT_ID: {"disabled_toolsets": ["memory"]}})
        enabled, disabled = _resolve(config, _source(CHAT_ID))
        agent_kwargs = {"enabled_toolsets": enabled, "disabled_toolsets": disabled}
        config["platforms"]["telegram"]["channel_overrides"][CHAT_ID] = {
            "disabled_toolsets": ["debugging"]}
        assert agent_kwargs["disabled_toolsets"] == ["memory"]


class TestDenyResolverDirect:
    """The pure resolver, exercised without the gateway runner."""

    def test_no_overrides(self):
        assert channel_override_deny_toolsets({}, chat_id="x") == []
        assert channel_override_deny_toolsets(None, chat_id="x") == []

    def test_deny_is_cumulative_across_matching_keys(self):
        raw = {
            "c1": {"disabled_toolsets": ["memory"]},
            "c1-parent": {"disabled_toolsets": ["web"]},
            "c2": {"disabled_toolsets": ["browser"]},
        }
        assert channel_override_deny_toolsets(
            raw, chat_id="c1", parent_id="c1-parent", chat_type="dm") == ["memory", "web"]

    def test_merge_deny_toolsets_dedupes_and_keeps_order(self):
        assert merge_deny_toolsets(["memory", "web"], ["web", "terminal"]) == ["memory", "web", "terminal"]

    def test_merge_deny_toolsets_accepts_json_string_global(self):
        # ``hermes config set`` persists a list as a JSON/Python-literal string; it must be parsed,
        # not treated as one literal name — otherwise the global deny silently filters nothing.
        assert merge_deny_toolsets("['memory']", []) == ["memory"]

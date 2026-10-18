"""
Tests for MEDIA tag extraction from tool results.

Verifies that MEDIA tags (e.g., from TTS tool) are only extracted from
messages in the CURRENT turn, not from the full conversation history.
This prevents voice messages from accumulating and being sent multiple
times per reply. (Regression test for #160)

Also covers #34608: a stale MEDIA: path emitted by an execute_code /
make_image tool several turns earlier must not leak onto a later
text-only reply, even when the path-based dedup set fails to capture it.
"""

import json
from unittest.mock import MagicMock

import pytest


class TestMediaExtraction:
    """Tests for MEDIA tag extraction from tool results."""

    def test_repairs_explicit_computer_use_media_path_from_json_result(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_0123456789abcdef0123456789abcdef.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        response = (
            "Here is the screenshot.\n"
            f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "capture", "function": {"name": "computer_use"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "capture",
                "content": json.dumps({"screenshot_path": canonical}),
            },
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f"Here is the screenshot.\nMEDIA:{canonical}"

    def test_repairs_explicit_path_from_multimodal_text_summary(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_fedcba9876543210fedcba9876543210.jpg"
        canonical = rf"D:\Hermes Data\cache\images\{capture_name}"
        response = f'MEDIA:"/Users/Alice/Hermes Data/cache/images/{capture_name}"'
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "tool_call_id": "capture",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "capture mode=screen 1920x1080\n"
                            f"  (shareable screenshot saved to {canonical})"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    },
                ],
            }
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f'MEDIA:"{canonical}"'

    def test_does_not_auto_attach_computer_use_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": canonical}),
            }
        ]

        assert (
            _repair_explicit_computer_use_media_paths("Done.", messages)
            == "Done."
        )

    def test_does_not_rewrite_unmatched_or_previous_turn_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        old_name = "computer_use_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png"
        current_name = "computer_use_cccccccccccccccccccccccccccccccc.png"
        old_canonical = rf"C:\cache\images\{old_name}"
        current_canonical = rf"C:\cache\images\{current_name}"
        history = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": old_canonical}),
            },
        ]
        current_turn = [
            {"role": "user", "content": "Send the current screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": current_canonical}),
            },
        ]
        response = f"MEDIA:/Users/Alice/cache/images/{old_name}"

        repaired = _repair_explicit_computer_use_media_paths(
            response,
            history + current_turn,
            history_offset=len(history),
        )

        assert repaired == response

    def test_malformed_json_result_fails_closed(self):
        """Truncated JSON must not repair to a doubled-backslash artifact.

        JSON escaping doubles backslashes; regex-scanning the raw string
        would yield ``C:\\\\Users\\\\...`` — a path that exists nowhere. When
        json.loads fails, the helper must yield nothing rather than rewrite
        the response to a corrupted path.
        """
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_dddddddddddddddddddddddddddddddd.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        payload = _json.dumps(
            {
                "summary": f"capture\n  (shareable screenshot saved to {canonical})",
                "screenshot_path": canonical,
            }
        )
        truncated = payload[:-5]  # starts with '{' but no longer parses
        response = f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        messages = [
            {"role": "tool", "name": "computer_use", "content": truncated},
        ]

        assert (
            _repair_explicit_computer_use_media_paths(response, messages)
            == response
        )

    def test_compression_fallback_slices_from_last_user_message(self):
        """When compression shrinks messages below history_offset, the repair
        recovers the current turn from the last user message — and fails
        closed (no rewrite) when no user message remains."""
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png"
        canonical = rf"C:\cache\images\{capture_name}"
        response = f"MEDIA:/cache/images/{capture_name}"
        current_turn = [
            {"role": "user", "content": "Send the screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": _json.dumps({"screenshot_path": canonical}),
            },
        ]

        # history_offset larger than the (compressed) message list forces the
        # fallback branch; the last-user slice still finds this turn's result.
        repaired = _repair_explicit_computer_use_media_paths(
            response, current_turn, history_offset=10
        )
        assert repaired == f"MEDIA:{canonical}"

        # No user message at all -> fail closed, nothing rewritten.
        no_user = [current_turn[1]]
        assert (
            _repair_explicit_computer_use_media_paths(
                response, no_user, history_offset=10
            )
            == response
        )

    def test_gateway_auto_append_ignores_media_examples_in_skill_docs(self):
        """Skill/documentation examples must not be appended as real attachments."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "How should I format gateway media?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_skill", "function": {"name": "skill_view"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_skill",
                "content": """
Recommended pattern:
```text
MEDIA:/absolute/path/to/image.png
```
Second message:
```text
caption
```
""",
            },
            {"role": "assistant", "content": "Use a standalone media message."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == []
        assert voice is False


    def test_collect_history_media_paths_includes_image_generate_json(self):
        """Regression for #46627: the history media-path collector must pick up
        image_generate JSON-payload paths (no MEDIA: tag), not just MEDIA:
        text tags. Otherwise, after a compression boundary the auto-append
        fallback rescans full history, finds the generated path absent from
        the dedup set, and re-emits the same MEDIA tag every turn.
        """
        from gateway.run import _collect_history_media_paths

        history = [
            {"role": "user", "content": "make a cat"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
            # A separate MEDIA: text tag from another tool, to confirm both shapes.
            {
                "role": "tool",
                "tool_call_id": "d",
                "content": "Saved MEDIA:/tmp/voice/note.ogg done",
            },
        ]
        paths = _collect_history_media_paths(history)
        assert "/tmp/gen/cat.png" in paths  # JSON-payload path (the bug)
        assert "/tmp/voice/note.ogg" in paths  # MEDIA: text path (already worked)

    def test_non_streaming_dedup_excludes_current_turn_tool_output(self):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/gen/old.png"
        current_path = "/tmp/gen/current.png"
        transcript = [
            {"role": "user", "content": "make the old image"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "make a new image"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "current", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "current",
                "content": f'{{"success": true, "image": "{current_path}"}}',
            },
            {"role": "assistant", "content": f"MEDIA:{current_path}"},
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    @pytest.mark.parametrize(
        "current_path",
        ["/tmp/tts/current.ogg", "/tmp/tts/already-delivered.ogg"],
    )
    def test_non_streaming_dedup_scopes_tts_paths_to_prior_turns(
        self, current_path
    ):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/tts/already-delivered.ogg"
        transcript = [
            {"role": "user", "content": "say the old message"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "say the current message"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tts", "function": {"name": "text_to_speech"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tts",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
            {
                "role": "assistant",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    def test_image_generate_not_reemitted_after_compression(self):
        """End-to-end of the #46627 fix: collect history paths, then the
        compression-fallback rescan (history_offset stale) must dedup the
        generated image against them — no re-emission."""
        from gateway.run import (
            _collect_auto_append_media_tags,
            _collect_history_media_paths,
        )

        history = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/dog.png"}',
            },
        ]
        history_paths = _collect_history_media_paths(history)

        # Simulate the post-compression fallback: history_offset is stale
        # (larger than the shrunken message list), so the collector rescans
        # the full list. With the dedup set populated, the already-delivered
        # image must NOT be re-emitted.
        tags, _ = _collect_auto_append_media_tags(
            history, history_offset=9999, history_media_paths=history_paths
        )
        assert tags == [], f"generated image re-emitted after compression: {tags}"


    
    
    




class TestBridgeDispatchedProducer:
    """Deferred/plugin tools are invoked through the generic ``tool_call`` bridge.

    The assistant tool-call row then only knows the bridge name, while the tool-result row
    keeps the real producer (``agent.tool_dispatch_helpers.make_tool_result_message`` sets
    ``name`` and ``tool_name``). MEDIA auto-append must judge the real producer, or a
    narrowly allowlisted plugin producer is silently dropped whenever the model omits the
    directive from its final reply.
    """

    GIF_PATH = "/tmp/giphy/turtle.gif"

    @staticmethod
    def _giphy_result(path: str) -> str:
        return json.dumps(
            {
                "ok": True,
                "media_path": path,
                "media_directive": f"MEDIA:{path}",
                "delivery_instruction": "Copy media_directive exactly into the final response.",
            }
        )

    def _messages(self, producer: str, path: str, *, row_name: bool = True):
        result_row = {
            "role": "tool",
            "tool_call_id": "call_bridge",
            "content": self._giphy_result(path),
        }
        if row_name:
            result_row["name"] = producer
            result_row["tool_name"] = producer
        return [
            {"role": "user", "content": "Send a turtle GIF"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_bridge", "function": {"name": "tool_call"}}],
            },
            result_row,
            {"role": "assistant", "content": "Here it is."},
        ]

    def test_bridge_call_appends_exactly_one_media_tag(self):
        from gateway.run_turn_runner import TurnRunner

        messages = self._messages("giphy_gif_search", self.GIF_PATH)
        runner = object.__new__(TurnRunner)  # _append_auto_media_tags touches no instance state

        rendered = runner._append_auto_media_tags(
            "Here it is.", {"messages": messages}, [], set()
        )

        assert rendered.count("MEDIA:") == 1, rendered
        assert f"MEDIA:{self.GIF_PATH}" in rendered

    def test_bridge_producer_does_not_duplicate_model_echo(self):
        """The model following its prompt ("copy media_directive verbatim") must not yield a
        second copy of the same attachment from auto-append."""
        from gateway.run_turn_runner import TurnRunner

        messages = self._messages("giphy_gif_search", self.GIF_PATH)
        runner = object.__new__(TurnRunner)

        echoed = runner._append_auto_media_tags(
            f"Here it is.\nMEDIA:{self.GIF_PATH}", {"messages": messages}, [], set()
        )

        assert echoed.count("MEDIA:") == 1, echoed
        assert echoed == f"Here it is.\nMEDIA:{self.GIF_PATH}"

    def test_direct_producer_call_still_resolves_from_call_row(self):
        """A direct call carries no name on the result row; the call-id map still resolves it."""
        from gateway.run import _collect_auto_append_media_tags

        messages = self._messages("giphy_gif_search", self.GIF_PATH, row_name=False)
        messages[1]["tool_calls"][0]["function"]["name"] = "giphy_gif_search"

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)

        assert tags == [f"MEDIA:{self.GIF_PATH}"]
        assert voice is False

    def test_bridge_dispatched_non_producer_is_dropped(self):
        """The allowlist stays a documented producer list: documentation examples from any
        other bridge-dispatched tool are never auto-attached."""
        from gateway.run import _collect_auto_append_media_tags

        messages = self._messages("tenor_gif_search", "/tmp/giphy/example_only.gif")

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)

        assert tags == []
        assert voice is False

    def test_bridge_producer_is_deduped_against_history(self):
        """A bridge-produced path already delivered in a prior turn must not be re-sent."""
        from gateway.run import _collect_auto_append_media_tags

        messages = self._messages("giphy_gif_search", self.GIF_PATH)

        tags, _voice = _collect_auto_append_media_tags(
            messages, history_offset=0, history_media_paths={self.GIF_PATH}
        )

        assert tags == []

    def test_core_producer_through_bridge_also_resolves(self):
        """The seam is producer-neutral: a core allowlisted producer bridged as ``tool_call``
        behaves like a direct call. Without the plugin no giphy row exists, so the only new
        allowlist member is unreachable — the lookup fix is what changes here, not the set."""
        from gateway.run import _collect_auto_append_media_tags

        row = {
            "role": "tool",
            "tool_call_id": "call_bridge",
            "name": "text_to_speech",
            "tool_name": "text_to_speech",
            "content": "MEDIA:/tmp/tts/hello.ogg\n[[audio_as_voice]]",
        }
        messages = [
            {"role": "user", "content": "say hello"},
            {"role": "assistant", "tool_calls": [{"id": "call_bridge", "function": {"name": "tool_call"}}]},
            row,
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)

        assert tags == ["MEDIA:/tmp/tts/hello.ogg"]
        assert voice is True

    def test_connector_and_delegation_results_never_append(self):
        """Connector batches and delegated subagent results may echo MEDIA: text; they are not
        producers, so they never append and can never duplicate the real producer's tag."""
        from gateway.run import _collect_auto_append_media_tags

        echo = json.dumps({"ok": True, "note": "attachment was MEDIA:/tmp/giphy/turtle.gif"})
        messages = [
            {"role": "user", "content": "Send a turtle GIF"},
            {"role": "tool", "tool_call_id": "c1", "name": "connector__gmail__send_email",
             "tool_name": "connector__gmail__send_email", "content": echo},
            {"role": "tool", "tool_call_id": "c2", "name": "delegate_task",
             "tool_name": "delegate_task", "content": echo},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)

        assert tags == []
        assert voice is False

    def test_producer_allowlist_is_the_documented_closed_set(self):
        """No pattern, no prefix rule: a producer is eligible only when named here."""
        from gateway.run import _AUTO_APPEND_MEDIA_TOOL_NAMES

        assert _AUTO_APPEND_MEDIA_TOOL_NAMES == {
            "text_to_speech",
            "text_to_speech_tool",
            "image_generate",
            "giphy_gif_search",
        }


class TestEffectiveToolName:
    """The shared producer seam used by MEDIA auto-append, history dedup and path repair."""

    def test_prefers_result_row_name_over_bridge_call_name(self):
        from gateway.media_repair import effective_tool_name

        msg = {
            "role": "tool",
            "tool_call_id": "call_bridge",
            "name": "tool_call",
            "tool_name": "giphy_gif_search",
            "content": "{}",
        }

        assert effective_tool_name(msg, {"call_bridge": "tool_call"}) == "giphy_gif_search"

    def test_falls_back_to_call_id_map(self):
        from gateway.media_repair import effective_tool_name

        msg = {"role": "tool", "tool_call_id": "call_tts", "content": "{}"}

        assert effective_tool_name(msg, {"call_tts": "text_to_speech"}) == "text_to_speech"

    def test_unknown_row_and_unknown_call_id_yield_empty(self):
        from gateway.media_repair import effective_tool_name

        assert effective_tool_name({"role": "tool", "content": "{}"}, {"other": "x"}) == ""
        assert effective_tool_name({"role": "tool", "content": "{}"}, None) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

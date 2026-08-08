from unittest.mock import MagicMock

import pytest

from pr_agent.algo.types import FilePatchInfo
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.tools.github_suggestion_dedup import finding_fingerprint, marker_for
from pr_agent.tools.pr_code_suggestions import PRCodeSuggestions


def _make_tool(git_provider=None):
    tool = PRCodeSuggestions.__new__(PRCodeSuggestions)
    tool.git_provider = git_provider or MagicMock()
    tool.progress_response = None
    return tool


def _valid_suggestion(**overrides):
    suggestion = {
        "one_sentence_summary": "Avoid duplicated work",
        "label": "maintainability",
        "relevant_file": "app.py",
        "relevant_lines_start": 1,
        "relevant_lines_end": 1,
        "suggestion_content": "Use the shared helper.",
        "existing_code": "old()",
        "improved_code": "new()",
    }
    suggestion.update(overrides)
    return suggestion


def test_prepare_pr_code_suggestions_filters_duplicates_and_missing_required_fields():
    tool = _make_tool()
    prediction = """
code_suggestions:
  - one_sentence_summary: Avoid duplicated work
    label: maintainability
    relevant_file: app.py
    suggestion_content: Use the shared helper.
    existing_code: old()
    improved_code: new()
  - one_sentence_summary: Avoid duplicated work
    label: maintainability
    relevant_file: app.py
    suggestion_content: Duplicate summary.
    existing_code: old()
    improved_code: newer()
  - one_sentence_summary: Missing label
    relevant_file: app.py
    suggestion_content: Missing label should be skipped.
    existing_code: old()
    improved_code: new()
"""

    data = tool._prepare_pr_code_suggestions(prediction)

    assert len(data["code_suggestions"]) == 1
    assert data["code_suggestions"][0]["one_sentence_summary"] == "Avoid duplicated work"
    assert data["code_suggestions"][0]["improved_code"] == "new()"


def test_prepare_pr_code_suggestions_renames_critical_label_when_focusing_only_on_problems():
    settings = get_settings()
    original_focus = settings.get("pr_code_suggestions.focus_only_on_problems", False)
    settings.set("pr_code_suggestions.focus_only_on_problems", True)
    tool = _make_tool()
    prediction = """
code_suggestions:
  - one_sentence_summary: Fix unsafe behavior
    label: critical issue
    relevant_file: app.py
    suggestion_content: Guard this path.
    existing_code: old()
    improved_code: new()
"""

    try:
        data = tool._prepare_pr_code_suggestions(prediction)

        assert data["code_suggestions"][0]["label"] == "possible issue"
    finally:
        settings.set("pr_code_suggestions.focus_only_on_problems", original_focus)


@pytest.mark.asyncio
async def test_analyze_self_reflection_response_merges_scores_and_zeroes_invalid_ranges():
    git_provider = MagicMock()
    git_provider.get_diff_files.return_value = []
    tool = _make_tool(git_provider)
    settings = get_settings()
    original_publish_output = settings.config.publish_output
    settings.config.publish_output = False
    suggestion = _valid_suggestion()
    suggestion.pop("relevant_lines_start")
    suggestion.pop("relevant_lines_end")
    data = {"code_suggestions": [suggestion]}
    response_reflect = """
code_suggestions:
  - suggestion_score: 9
    why: Great suggestion, but line range is missing.
    relevant_lines_start: -1
    relevant_lines_end: -1
"""

    try:
        await tool.analyze_self_reflection_response(data, response_reflect)

        assert data["code_suggestions"][0]["score"] == 0
        assert data["code_suggestions"][0]["score_why"] == "Great suggestion, but line range is missing."
        assert data["code_suggestions"][0]["relevant_lines_start"] == -1
        assert data["code_suggestions"][0]["relevant_lines_end"] == -1
    finally:
        settings.config.publish_output = original_publish_output


def test_dedent_code_matches_target_file_indentation():
    git_provider = MagicMock()
    git_provider.diff_files = [
        FilePatchInfo(
            base_file="",
            head_file="def f():\n    return old()\n",
            patch="",
            filename="app.py",
        )
    ]
    tool = _make_tool(git_provider)

    assert tool.dedent_code("app.py", 2, "return new()") == "    return new()"


@pytest.mark.asyncio
async def test_push_inline_code_suggestions_falls_back_to_individual_publish_calls():
    git_provider = MagicMock()
    git_provider.diff_files = [
        FilePatchInfo(
            base_file="",
            head_file="def f():\n    return old()\n",
            patch="",
            filename="app.py",
        ),
        FilePatchInfo(
            base_file="",
            head_file="def work():\n    return old_worker()\n",
            patch="",
            filename="worker.py",
        ),
    ]
    git_provider.publish_code_suggestions.side_effect = [False, True, True]
    tool = _make_tool(git_provider)
    data = {"code_suggestions": [
        _valid_suggestion(
            relevant_lines_start=2,
            relevant_lines_end=2,
            score=8,
        ),
        _valid_suggestion(
            relevant_file="worker.py",
            relevant_lines_start=2,
            relevant_lines_end=2,
            existing_code="return old_worker()",
            improved_code="return new_worker()",
            suggestion_content="Keep the worker result fresh.",
        ),
    ]}

    await tool.push_inline_code_suggestions(data)

    assert git_provider.publish_code_suggestions.call_count == 3
    batch_call = git_provider.publish_code_suggestions.call_args_list[0].args[0]
    first_retry = git_provider.publish_code_suggestions.call_args_list[1].args[0]
    second_retry = git_provider.publish_code_suggestions.call_args_list[2].args[0]
    assert len(batch_call) == 2
    assert first_retry == [batch_call[0]]
    assert second_retry == [batch_call[1]]
    assert first_retry[0]["relevant_file"] == "app.py"
    assert first_retry[0]["relevant_lines_start"] == 2
    assert first_retry[0]["relevant_lines_end"] == 2
    assert "```suggestion\n    new()" in first_retry[0]["body"]
    assert second_retry[0]["relevant_file"] == "worker.py"
    assert second_retry[0]["relevant_lines_start"] == 2
    assert second_retry[0]["relevant_lines_end"] == 2
    assert "```suggestion\n    return new_worker()" in second_retry[0]["body"]


def test_persistent_update_survives_progress_cleanup_failure():
    """A failing progress-note cleanup must not abort the persistent update:
    if the cleanup error propagated, the caller would fall back to publishing
    a new suggestions thread, re-creating the duplicate-thread bug."""
    initial_header = "## PR Code Suggestions"
    existing = MagicMock()
    existing.body = f"{initial_header}\n<!-- aaa1111 -->\n<table>old suggestions</table>"
    provider = MagicMock()
    provider.get_issue_comments.return_value = [existing]
    provider.get_comment_url.return_value = "https://example.test/comment/1"
    provider.get_latest_commit_url.return_value = "https://example.test/commit/deadbee"
    # First edit updates the persistent comment and succeeds; the second edit
    # (re-labelling the progress note before deletion) fails.
    provider.edit_comment.side_effect = [None, RuntimeError("cleanup failed")]
    progress_note = MagicMock()

    result = PRCodeSuggestions.publish_persistent_comment_with_history(
        provider, f"{initial_header}\n<table>new suggestions</table>", initial_header,
        update_header=False, name="suggestions", final_update_message=False,
        progress_response=progress_note)

    assert result is existing
    assert provider.edit_comment.call_count == 2
    provider.remove_comment.assert_not_called()
    provider.publish_comment.assert_not_called()


def _make_github_provider_for_dedup(history):
    provider = GithubProvider.__new__(GithubProvider)
    provider.diff_files = [
        FilePatchInfo(base_file="", head_file="return old()\n", patch="", filename="app.py")
    ]
    provider.get_code_suggestion_history = MagicMock(return_value=history)
    provider.publish_code_suggestions = MagicMock(return_value=True)
    return provider


@pytest.mark.asyncio
async def test_inline_dedup_adds_marker_and_reuses_history_for_dual_publication_path():
    history = {"comments": [], "thread_states": {}, "bot_login": "pr-agent[bot]"}
    provider = _make_github_provider_for_dedup(history)
    tool = _make_tool(provider)
    data = {"code_suggestions": [_valid_suggestion(score=8)]}
    settings = get_settings()
    previous = settings.pr_code_suggestions.get("deduplicate_suggestions", True)
    previous_threshold = settings.pr_code_suggestions.dual_publishing_score_threshold
    settings.pr_code_suggestions.deduplicate_suggestions = True
    settings.pr_code_suggestions.dual_publishing_score_threshold = 1
    try:
        # Exercise the direct committable path, followed by the table mode's dual-publishing path.
        await tool.push_inline_code_suggestions(data)
        await tool.dual_publishing(data)
    finally:
        settings.pr_code_suggestions.deduplicate_suggestions = previous
        settings.pr_code_suggestions.dual_publishing_score_threshold = previous_threshold

    assert provider.get_code_suggestion_history.call_count == 1
    assert provider.publish_code_suggestions.call_count == 2
    assert "<!-- pr-agent-finding:" in provider.publish_code_suggestions.call_args.args[0][0]["body"]


@pytest.mark.asyncio
async def test_inline_dedup_api_failure_fails_open():
    provider = _make_github_provider_for_dedup({})
    provider.get_code_suggestion_history.side_effect = RuntimeError("GraphQL unavailable")
    tool = _make_tool(provider)
    settings = get_settings()
    previous = settings.pr_code_suggestions.get("deduplicate_suggestions", True)
    settings.pr_code_suggestions.deduplicate_suggestions = True
    try:
        await tool.push_inline_code_suggestions({"code_suggestions": [_valid_suggestion()]})
        await tool.push_inline_code_suggestions({"code_suggestions": [_valid_suggestion()]})
    finally:
        settings.pr_code_suggestions.deduplicate_suggestions = previous

    assert provider.get_code_suggestion_history.call_count == 1
    assert provider.publish_code_suggestions.call_count == 2
    assert "pr-agent-finding" not in provider.publish_code_suggestions.call_args.args[0][0]["body"]


@pytest.mark.asyncio
async def test_inline_dedup_configuration_disabled_skips_retrieval():
    provider = _make_github_provider_for_dedup({})
    tool = _make_tool(provider)
    settings = get_settings()
    previous = settings.pr_code_suggestions.get("deduplicate_suggestions", True)
    settings.pr_code_suggestions.deduplicate_suggestions = False
    try:
        await tool.push_inline_code_suggestions({"code_suggestions": [_valid_suggestion()]})
    finally:
        settings.pr_code_suggestions.deduplicate_suggestions = previous

    provider.get_code_suggestion_history.assert_not_called()
    assert provider.publish_code_suggestions.call_count == 1


@pytest.mark.asyncio
async def test_inline_dedup_suppresses_empty_output_when_configured():
    item = _valid_suggestion()
    history = {
        "comments": [{
            "id": 1,
            "in_reply_to_id": None,
            "body": marker_for(finding_fingerprint(item)),
            "path": item["relevant_file"],
            "diff_hunk": item["existing_code"],
            "author_login": "pr-agent[bot]",
            "author_association": "NONE",
        }],
        "thread_states": {1: {"resolved": False, "outdated": False}},
        "bot_login": "pr-agent[bot]",
    }
    provider = _make_github_provider_for_dedup(history)
    tool = _make_tool(provider)
    settings = get_settings()
    previous = settings.pr_code_suggestions.publish_output_no_suggestions
    settings.pr_code_suggestions.publish_output_no_suggestions = False
    try:
        await tool.push_inline_code_suggestions({"code_suggestions": [item]})
    finally:
        settings.pr_code_suggestions.publish_output_no_suggestions = previous

    provider.publish_code_suggestions.assert_not_called()

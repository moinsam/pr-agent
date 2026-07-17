import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.tools.github_suggestion_dedup import (filter_duplicate_suggestions,
                                                    finding_fingerprint,
                                                    marker_for)


def suggestion(**overrides):
    item = {
        "one_sentence_summary": "Avoid duplicated work",
        "label": "maintainability",
        "suggestion_content": "Use the shared helper.",
        "relevant_file": "src/app.py",
        "existing_code": "return old()",
        "improved_code": "return shared()",
    }
    item.update(overrides)
    return item


def comment(item, *, comment_id=1, body=None, path=None, diff_hunk=None, author="pr-agent[bot]",
            reply_to=None, association="NONE"):
    return {
        "id": comment_id,
        "in_reply_to_id": reply_to,
        "body": body if body is not None else marker_for(finding_fingerprint(item)),
        "path": path if path is not None else item["relevant_file"],
        "diff_hunk": diff_hunk if diff_hunk is not None else item["existing_code"],
        "author_login": author,
        "author_association": association,
    }


def apply_filter(items, comments, *, states=None, **options):
    history = {"comments": comments, "thread_states": states or {}, "bot_login": "pr-agent[bot]"}
    defaults = {"bot_logins": [], "include_resolved": True, "honor_decisions": True,
                "similarity_threshold": 0.75}
    defaults.update(options)
    return filter_duplicate_suggestions(items, history, **defaults)


def test_exact_fingerprint_suppresses_unresolved_and_moved_line():
    item = suggestion(relevant_lines_start=42)
    assert apply_filter([item], [comment(item)]) == []


def test_same_message_in_different_files_is_not_suppressed():
    old = suggestion(relevant_file="src/other.py")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_paths_that_differ_only_by_case_are_distinct():
    old = suggestion(relevant_file="src/App.py")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_distinct_findings_in_same_file_are_not_suppressed():
    old = suggestion(suggestion_content="Close the socket.", existing_code="socket.open()")
    assert len(apply_filter([suggestion()], [comment(old)])) == 1


def test_resolved_finding_obeys_configuration():
    item = suggestion()
    states = {1: {"resolved": True, "outdated": True}}
    assert apply_filter([item], [comment(item)], states=states) == []
    assert len(apply_filter([item], [comment(item)], states=states, include_resolved=False)) == 1


def test_outdated_unresolved_finding_obeys_configuration():
    item = suggestion()
    states = {1: {"resolved": False, "outdated": True}}
    assert apply_filter([item], [comment(item)], states=states) == []
    assert len(apply_filter([item], [comment(item)], states=states, include_resolved=False)) == 1


@pytest.mark.parametrize("command", ["ignore", "accepted-risk"])
def test_maintainer_decisions_are_case_insensitive_and_allow_whitespace(command):
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="maintainer", association="MEMBER",
                    body=f"  PR-Agent: {command.upper()}  ")
    assert apply_filter([item], [comment(item), reply], include_resolved=False) == []


def test_fixed_allows_reintroduced_finding():
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, author="maintainer", association="OWNER",
                    body="pr-agent: fixed")
    assert len(apply_filter([item], [comment(item), reply])) == 1


def test_bot_authored_command_is_ignored():
    item = suggestion()
    reply = comment(item, comment_id=2, reply_to=1, body="pr-agent: ignore")
    assert len(apply_filter([item], [comment(item), reply], states={1: {"resolved": True}},
                            include_resolved=False)) == 1


def test_material_code_change_changes_fingerprint_and_is_not_suppressed():
    old = suggestion(existing_code="return old()")
    new = suggestion(existing_code="return old(value, strict=True)")
    assert len(apply_filter([new], [comment(old)])) == 1


def test_legacy_comment_without_marker_is_matched_without_generated_summary():
    item = suggestion(one_sentence_summary="A summary that was never rendered in the old comment")
    legacy_body = "**Suggestion:** Use the shared helper. [maintainability]\n```suggestion\nreturn shared()\n```"
    assert apply_filter([item], [comment(item, body=legacy_body)]) == []


def test_legacy_comment_matches_code_inside_larger_diff_hunk_at_default_threshold():
    item = suggestion(one_sentence_summary="Different generated summary")
    legacy_body = ("**Suggestion:** Use the shared helper. [maintainability, importance: 8]\n"
                   "```suggestion\nreturn shared()\n```")
    diff_hunk = "@@ -10,3 +10,3 @@ def run():\n context()\n-return old()\n+return shared()"
    assert apply_filter([item], [comment(item, body=legacy_body, diff_hunk=diff_hunk)],
                        similarity_threshold=0.88) == []


def test_malformed_marker_is_ignored():
    item = suggestion()
    malformed = comment(item, body="<!-- pr-agent-finding: not-a-hash -->")
    assert len(apply_filter([item], [malformed])) == 1


def test_configured_bot_logins_restrict_roots():
    item = suggestion()
    assert len(apply_filter([item], [comment(item)], bot_logins=["another-bot"])) == 1


class PaginatedComments:
    def __init__(self, comments):
        self.comments = comments

    def __iter__(self):
        yield from self.comments


def _raw_comment(comment_id):
    raw = {"id": comment_id, "body": "body", "path": "src/app.py", "diff_hunk": "return old()",
           "user": {"login": "pr-agent[bot]"}, "author_association": "NONE"}
    return SimpleNamespace(id=comment_id, raw_data=raw)


def test_provider_retrieves_paginated_rest_comments_and_graphql_threads():
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 7
    provider.github_user_id = "pr-agent[bot]"
    provider.pr = SimpleNamespace(get_comments=lambda: PaginatedComments([_raw_comment(1), _raw_comment(2)]))
    responses = [
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [{"isResolved": False, "isOutdated": False,
                       "comments": {"nodes": [{"databaseId": 1}]}}],
            "pageInfo": {"hasNextPage": True, "endCursor": "next"}}}}}},
        {"data": {"repository": {"pullRequest": {"reviewThreads": {
            "nodes": [{"isResolved": True, "isOutdated": True,
                       "comments": {"nodes": [{"databaseId": 2}]}}],
            "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}},
    ]
    requester = MagicMock()
    requester.requestJson.side_effect = [(200, {}, json.dumps(response)) for response in responses]
    provider.github_client = SimpleNamespace(_Github__requester=requester)

    history = provider.get_code_suggestion_history()

    assert [item["id"] for item in history["comments"]] == [1, 2]
    assert history["thread_states"][2] == {"resolved": True, "outdated": True}
    assert requester.requestJson.call_count == 2


def test_provider_graphql_error_is_raised_for_caller_fail_open():
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "owner/repo"
    provider.pr_num = 7
    provider.github_user_id = "pr-agent[bot]"
    provider.pr = SimpleNamespace(get_comments=lambda: [])
    requester = MagicMock()
    requester.requestJson.return_value = (200, {}, json.dumps({"errors": [{"message": "denied"}]}))
    provider.github_client = SimpleNamespace(_Github__requester=requester)
    with pytest.raises(RuntimeError):
        provider.get_code_suggestion_history()


def test_provider_rest_error_is_raised_for_caller_fail_open():
    provider = GithubProvider.__new__(GithubProvider)
    provider.pr = SimpleNamespace(get_comments=MagicMock(side_effect=RuntimeError("REST unavailable")))
    with pytest.raises(RuntimeError):
        provider.get_code_suggestion_history()

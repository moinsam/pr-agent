import hashlib
import posixpath
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


FINDING_MARKER_RE = re.compile(r"<!--\s*pr-agent-finding:\s*([0-9a-f]{64})\s*-->", re.IGNORECASE)
MALFORMED_FINDING_MARKER_RE = re.compile(r"<!--\s*pr-agent-finding:", re.IGNORECASE)
DECISION_RE = re.compile(r"^\s*pr-agent:\s*(ignore|accepted-risk|fixed)\s*$", re.IGNORECASE | re.MULTILINE)
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def normalize_path(path: str) -> str:
    path = str(path or "").strip().strip("`").replace("\\", "/")
    path = posixpath.normpath(path).lstrip("/")
    if path in {"", "."} or path == ".." or path.startswith("../"):
        return ""
    return path


def normalize_text(value: str) -> str:
    value = re.sub(r"<!--.*?-->", " ", str(value or ""), flags=re.DOTALL)
    value = re.sub(r"```(?:suggestion)?|```", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE)
    return " ".join(value.split())[:4000]


def normalize_code(value: str) -> str:
    lines = []
    for line in str(value or "").replace("\r\n", "\n").splitlines():
        line = re.sub(r"^\s*[-+ ]", "", line).strip()
        if line and not line.startswith("@@"):
            lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)[-4000:]


def finding_parts(suggestion: dict[str, Any]) -> tuple[str, str, str]:
    path = normalize_path(suggestion.get("relevant_file", ""))
    semantic = normalize_text(" ".join(str(suggestion.get(key, "")) for key in (
        "one_sentence_summary", "label", "suggestion_content", "relevant_symbol", "symbol"
    )))
    context = normalize_code(suggestion.get("existing_code", "") or suggestion.get("nearby_code", ""))
    return path, semantic, context


def legacy_finding_parts(suggestion: dict[str, Any]) -> tuple[str, str, str]:
    """Return fields represented in pre-marker inline comments."""
    path = normalize_path(suggestion.get("relevant_file", ""))
    semantic = normalize_text(" ".join(str(suggestion.get(key, "")) for key in (
        "suggestion_content", "label", "relevant_symbol", "symbol"
    )))
    context = normalize_code(suggestion.get("existing_code", "") or suggestion.get("nearby_code", ""))
    return path, semantic, context


def finding_fingerprint(suggestion: dict[str, Any]) -> str:
    return hashlib.sha256("\x1f".join(finding_parts(suggestion)).encode("utf-8")).hexdigest()


def marker_for(fingerprint: str) -> str:
    return f"<!-- pr-agent-finding: {fingerprint} -->"


def parse_marker(body: str) -> str | None:
    match = FINDING_MARKER_RE.search(str(body or ""))
    return match.group(1).lower() if match else None


def parse_decision(body: str) -> str | None:
    match = DECISION_RE.search(str(body or ""))
    return match.group(1).lower() if match else None


@dataclass(frozen=True)
class PriorFinding:
    fingerprint: str | None
    path: str
    semantic: str
    context: str
    resolved: bool
    outdated: bool
    decision: str | None


def _legacy_parts(comment: dict[str, Any]) -> tuple[str, str, str]:
    body = str(comment.get("body", ""))
    semantic_body = body.split("```suggestion", 1)[0]
    semantic_body = re.sub(r"^\s*\*\*Suggestion:\*\*\s*", "", semantic_body, flags=re.IGNORECASE)
    semantic_body = re.sub(r",\s*importance:\s*\d+(?:\.\d+)?", "", semantic_body, flags=re.IGNORECASE)
    return (
        normalize_path(comment.get("path", "")),
        normalize_text(semantic_body),
        normalize_code(comment.get("diff_hunk", "") or comment.get("nearby_code", "")),
    )


def index_prior_findings(history: dict[str, Any], bot_logins: list[str], honor_decisions: bool) -> list[PriorFinding]:
    comments = history.get("comments", [])
    configured_logins = {login.casefold() for login in bot_logins if login}
    authenticated_login = str(history.get("bot_login", "")).casefold()
    recognized_logins = configured_logins or ({authenticated_login} if authenticated_login else set())
    by_root: dict[int, list[dict[str, Any]]] = {}
    for comment in comments:
        comment_id = comment.get("id")
        root_id = comment.get("in_reply_to_id") or comment_id
        if isinstance(root_id, int):
            by_root.setdefault(root_id, []).append(comment)

    findings = []
    thread_states = history.get("thread_states", {})
    for root_id, thread_comments in by_root.items():
        root = next((comment for comment in thread_comments if comment.get("id") == root_id), None)
        if not root or str(root.get("author_login", "")).casefold() not in recognized_logins:
            continue
        body = str(root.get("body", ""))
        marker = parse_marker(body)
        if marker is None and MALFORMED_FINDING_MARKER_RE.search(body):
            continue
        decision = None
        if honor_decisions:
            for reply in sorted(thread_comments, key=lambda item: item.get("id", 0)):
                if reply is root or str(reply.get("author_login", "")).casefold() in recognized_logins:
                    continue
                if str(reply.get("author_association", "")).upper() not in MAINTAINER_ASSOCIATIONS:
                    continue
                decision = parse_decision(reply.get("body", "")) or decision
        path, semantic, context = _legacy_parts(root)
        state = thread_states.get(root_id, {})
        findings.append(PriorFinding(marker, path, semantic, context, bool(state.get("resolved")),
                                     bool(state.get("outdated")), decision))
    return findings


def _legacy_equivalent(parts: tuple[str, str, str], prior: PriorFinding, threshold: float) -> bool:
    path, semantic, context = parts
    if not path or path != prior.path or not semantic or not prior.semantic:
        return False
    semantic_tokens = set(semantic.split())
    prior_tokens = set(prior.semantic.split())
    token_ratio = len(semantic_tokens & prior_tokens) / max(len(semantic_tokens), len(prior_tokens))
    semantic_ratio = max(SequenceMatcher(None, semantic, prior.semantic, autojunk=False).ratio(), token_ratio)
    context_ratio = (SequenceMatcher(None, context, prior.context, autojunk=False).ratio()
                     if context and prior.context else 0.0)
    context_match = context_ratio >= threshold or context in prior.context or prior.context in context
    return semantic_ratio >= threshold and (context_match or not context or not prior.context)


def filter_duplicate_suggestions(suggestions: list[dict[str, Any]], history: dict[str, Any], *,
                                 bot_logins: list[str], include_resolved: bool,
                                 honor_decisions: bool, similarity_threshold: float) -> list[dict[str, Any]]:
    threshold = min(1.0, max(0.0, float(similarity_threshold)))
    prior_findings = index_prior_findings(history, bot_logins, honor_decisions)
    filtered = []
    for suggestion in suggestions:
        fingerprint = finding_fingerprint(suggestion)
        legacy_parts = legacy_finding_parts(suggestion)
        suppress = False
        for prior in prior_findings:
            equivalent = (prior.fingerprint == fingerprint if prior.fingerprint
                          else _legacy_equivalent(legacy_parts, prior, threshold))
            if not equivalent:
                continue
            # A generated finding means the problematic context is present. A previous "fixed" decision therefore
            # does not suppress a reintroduced finding; ignore and accepted-risk decisions remain durable.
            durable_decision = prior.decision in {"ignore", "accepted-risk"}
            historical_thread = prior.resolved or prior.outdated
            suppress = durable_decision or (
                prior.decision != "fixed" and (include_resolved or not historical_thread)
            )
            if suppress:
                break
        if not suppress:
            item = dict(suggestion)
            item["finding_fingerprint"] = fingerprint
            filtered.append(item)
    return filtered

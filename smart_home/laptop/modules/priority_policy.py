from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import config

# Chuẩn hóa bộ trạng thái ưu tiên/cơ chế quyết định
DECISION_MODE_SINGLE_USER = "single_user"
DECISION_MODE_MULTI_USER = "multi_user"
DECISION_MODE_MANUAL_OVERRIDE = "manual_override"
DECISION_MODE_FALLBACK = "fallback"

PRIORITY_REASON_HOST = "host_priority"
PRIORITY_REASON_GUEST_ONLY = "guest_only"
PRIORITY_REASON_POLICY_MAJORITY = "policy_majority"
PRIORITY_REASON_POLICY_OLDEST_SEEN = "policy_oldest_seen"

MULTI_USER_POLICIES = {"host_priority", "dominant_host", "oldest_seen", "majority_vote", "priority_rank"}


@dataclass(frozen=True)
class PriorityResolution:
    user_id: str
    dominant_user_id: str
    active_users: tuple[str, ...]
    decision_mode: str
    priority_reason: str


def normalize_decision_mode(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {DECISION_MODE_SINGLE_USER, DECISION_MODE_MULTI_USER, DECISION_MODE_MANUAL_OVERRIDE, DECISION_MODE_FALLBACK}:
        return text
    if text in {"multi_user_priority", "multi_user_host_priority", "multi_user_majority_vote", "multi_user_oldest_seen"}:
        return DECISION_MODE_MULTI_USER
    if text in {"manual", "override", "manual_override"}:
        return DECISION_MODE_MANUAL_OVERRIDE
    return DECISION_MODE_FALLBACK if text else DECISION_MODE_SINGLE_USER


def normalize_priority_reason(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {PRIORITY_REASON_HOST, PRIORITY_REASON_GUEST_ONLY, PRIORITY_REASON_POLICY_MAJORITY, PRIORITY_REASON_POLICY_OLDEST_SEEN}:
        return text
    if text in {"priority_map", "priority_map:dominant", "priority_map:user_a", "priority_map:user_b"}:
        return PRIORITY_REASON_HOST
    if text in {"majority_vote_policy", "majority_vote"}:
        return PRIORITY_REASON_POLICY_MAJORITY
    if text in {"oldest_seen_policy", "oldest_seen"}:
        return PRIORITY_REASON_POLICY_OLDEST_SEEN
    return text or PRIORITY_REASON_HOST


def map_scenario_type(decision_mode: str, source: str = "", target_user: str = "", override_flag: int = 0, review_reason: str = "", event_type: str = "", presence_type: str = "") -> str:
    decision_mode = normalize_decision_mode(decision_mode)
    source = (source or "").strip().lower()
    target_user = (target_user or "").strip().lower()
    review_reason = (review_reason or "").strip().lower()
    event_type = (event_type or "").strip().lower()
    presence_type = (presence_type or "").strip().lower()

    if override_flag or event_type in {"override_feedback", "manual_override"} or source in {"manual_override", "manual"} or decision_mode == DECISION_MODE_MANUAL_OVERRIDE:
        return "manual_override"
    if source in {"needs_review", "context_exception"} or "low_confidence" in review_reason or event_type in {"context_exception", "exception"}:
        return "context_exception"
    if decision_mode == DECISION_MODE_MULTI_USER or source == "face_recognition" or "profile" in event_type or "profile" in source or target_user in {"global_average"} or presence_type in {"multi_user", "single_known"}:
        return "profile_personalization"
    if source in {"auto", "shadow", "shadow_pir", "pir_motion", "pir_only", "pir_timeout_total_off"}:
        return "sequence_behavior"
    return "background_noise"


def resolve_priority(active_users: Iterable[str], current_user: str | None = None) -> PriorityResolution:
    priority_map = getattr(config, "USER_PRIORITY", {}) or {}
    users = []
    for user in active_users:
        if user is None:
            continue
        text = str(user).strip().lower()
        if text and text not in {"unknown"}:
            users.append(text)
    current_user = (current_user or "unknown").strip().lower() or "unknown"
    if current_user not in {"unknown", "global_average"} and current_user not in users:
        users.append(current_user)

    if not users:
        return PriorityResolution("unknown", "unknown", (), DECISION_MODE_FALLBACK, PRIORITY_REASON_HOST)

    ranked = sorted(users, key=lambda uid: (priority_map.get(uid, 999), uid))
    dominant = ranked[0]
    if len(users) == 1:
        return PriorityResolution(dominant, dominant, tuple(users), DECISION_MODE_SINGLE_USER, PRIORITY_REASON_HOST)

    host_candidates = [u for u in users if priority_map.get(u, 999) <= 2]
    if host_candidates:
        dominant = sorted(host_candidates, key=lambda uid: (priority_map.get(uid, 999), uid))[0]
        return PriorityResolution(dominant, dominant, tuple(users), DECISION_MODE_MULTI_USER, PRIORITY_REASON_HOST)

    if all(priority_map.get(u, 999) >= 3 for u in users):
        return PriorityResolution(dominant, dominant, tuple(users), DECISION_MODE_MULTI_USER, PRIORITY_REASON_GUEST_ONLY)

    return PriorityResolution(dominant, dominant, tuple(users), DECISION_MODE_MULTI_USER, PRIORITY_REASON_HOST)

import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

from modules.priority_policy import (
    DECISION_MODE_FALLBACK,
    DECISION_MODE_MANUAL_OVERRIDE,
    DECISION_MODE_MULTI_USER,
    DECISION_MODE_SINGLE_USER,
    PRIORITY_REASON_GUEST_ONLY,
    PRIORITY_REASON_HOST,
    PRIORITY_REASON_POLICY_MAJORITY,
    PRIORITY_REASON_POLICY_OLDEST_SEEN,
    normalize_decision_mode,
    normalize_priority_reason,
)
import config


@dataclass
class DeviceDecision:
    light: int
    fan: int
    confidence: float = 0.0
    source: str = "unknown"
    user_id: str = "unknown"
    shadow: bool = False
    pir_handled: bool = False
    active_users: tuple[str, ...] = ()
    dominant_user_id: str = "unknown"
    decision_mode: str = "single_user"
    priority_reason: str = ""


class DecisionCoordinator:
    """Bộ điều phối chung để giảm trùng logic main/web.

    Mục tiêu:
    - Chuẩn hóa quyết định AI -> lệnh thiết bị
    - Tránh web/main tự viết lại cùng một logic apply state
    - Giữ một điểm duy nhất để kiểm tra shadow/manual/pir
    """

    def __init__(self):
        self._lock = threading.Lock()

    def _normalize_user_id(self, value: str | None) -> str:
        text = (value or "unknown")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip().lower() or "unknown"
        if text in {"", "none", "null"}:
            return "unknown"
        if text == "guest":
            return "global_average"
        return text

    def normalize(self, result: dict, user_id: str, shadow: bool = False, pir_handled: bool = False,
                   active_users: Optional[Sequence[str]] = None, dominant_user: Optional[str] = None,
                   decision_mode: str = DECISION_MODE_SINGLE_USER, priority_reason: str = PRIORITY_REASON_HOST) -> DeviceDecision:
        normalized_active = tuple(
            self._normalize_user_id(u) for u in (active_users or []) if self._normalize_user_id(u) not in ("unknown",)
        )
        norm_user = self._normalize_user_id(user_id)
        return DeviceDecision(
            light=int(bool(result.get("light", 0))),
            fan=int(bool(result.get("fan", 0))),
            confidence=float(result.get("confidence", 0.0)),
            source=str(result.get("source", "unknown")),
            user_id=norm_user,
            shadow=shadow,
            pir_handled=pir_handled,
            active_users=normalized_active,
            dominant_user_id=self._normalize_user_id(dominant_user or norm_user),
            decision_mode=normalize_decision_mode(decision_mode),
            priority_reason=normalize_priority_reason(priority_reason),
        )

    def normalize_full(self, result: dict, user_id: str, shadow: bool = False, pir_handled: bool = False,
                       active_users: Optional[Sequence[str]] = None, dominant_user: Optional[str] = None,
                       decision_mode: str = DECISION_MODE_SINGLE_USER, priority_reason: str = PRIORITY_REASON_HOST) -> DeviceDecision:
        decision = self.normalize(
            result,
            user_id=user_id,
            shadow=shadow,
            pir_handled=pir_handled,
            active_users=active_users,
            dominant_user=dominant_user,
            decision_mode=decision_mode,
            priority_reason=priority_reason,
        )
        for key, value in result.items():
            if hasattr(decision, key):
                continue
            setattr(decision, key, value)
        for key in ("override_flag", "override_reason", "needs_review", "review_reason", "metadata", "raw_result"):
            if key in result:
                setattr(decision, key, result[key])
        return decision

    def should_execute(self, decision: DeviceDecision, manual_mode: bool) -> bool:
        return not manual_mode and not decision.shadow

    def resolve_conflict(self, decision: DeviceDecision, current_user: str, active_users: Optional[list[str]] = None) -> DeviceDecision:
        """Ưu tiên người dùng có thứ tự cao nhất trong phòng."""
        active_users = [self._normalize_user_id(u) for u in (active_users or [])]
        active_users = [u for u in active_users if u not in ("unknown",)]
        current_user = self._normalize_user_id(current_user)
        policy = getattr(config, "MULTI_USER_DECISION_POLICY", "host_priority").strip().lower()
        fallback_mode = getattr(config, "MULTI_USER_FALLBACK_MODE", "dominant_host").strip().lower()
        min_faces = max(2, int(getattr(config, "MULTI_USER_MIN_FACE_COUNT", 2)))
        priority_map = getattr(config, "USER_PRIORITY", {}) or {}

        if current_user not in ("unknown", "global_average"):
            if current_user not in active_users:
                active_users.append(current_user)
            if len(active_users) <= 1:
                decision.user_id = current_user
                decision.dominant_user_id = current_user
                decision.active_users = tuple(active_users)
                decision.decision_mode = DECISION_MODE_SINGLE_USER
                decision.priority_reason = PRIORITY_REASON_HOST
                return decision

        if not active_users:
            decision.user_id = current_user
            decision.dominant_user_id = current_user
            decision.active_users = ()
            decision.decision_mode = DECISION_MODE_FALLBACK
            decision.priority_reason = PRIORITY_REASON_HOST
            return decision

        ranked = sorted(active_users, key=lambda uid: (priority_map.get(uid, 999), uid))
        dominant = ranked[0] if ranked else decision.user_id

        if policy in {"host_priority", "dominant_host", "priority_rank"}:
            decision.user_id = dominant
            decision.dominant_user_id = dominant
            decision.active_users = tuple(active_users)
            decision.decision_mode = DECISION_MODE_MULTI_USER if len(active_users) >= min_faces else DECISION_MODE_SINGLE_USER
            decision.priority_reason = PRIORITY_REASON_HOST
            return decision

        if policy == "oldest_seen":
            decision.user_id = current_user if current_user in active_users else dominant
            decision.dominant_user_id = decision.user_id
            decision.active_users = tuple(active_users)
            decision.decision_mode = DECISION_MODE_MULTI_USER if len(active_users) >= min_faces else DECISION_MODE_SINGLE_USER
            decision.priority_reason = PRIORITY_REASON_POLICY_OLDEST_SEEN
            return decision

        if policy == "majority_vote":
            if len(active_users) >= min_faces:
                from collections import Counter

                counts = Counter(active_users)
                top_count = max(counts.values()) if counts else 0
                top_users = sorted([u for u, c in counts.items() if c == top_count], key=lambda uid: (priority_map.get(uid, 999), uid))
                vote_winner = top_users[0] if top_users else dominant
                if fallback_mode == "current_user" and current_user in active_users:
                    vote_winner = current_user
                decision.user_id = vote_winner
                decision.dominant_user_id = vote_winner
                decision.active_users = tuple(active_users)
                decision.decision_mode = DECISION_MODE_MULTI_USER
                decision.priority_reason = PRIORITY_REASON_POLICY_MAJORITY
                return decision

        decision.user_id = dominant
        decision.dominant_user_id = dominant
        decision.active_users = tuple(active_users)
        decision.decision_mode = DECISION_MODE_MULTI_USER if len(active_users) >= min_faces else DECISION_MODE_SINGLE_USER
        decision.priority_reason = PRIORITY_REASON_HOST
        return decision

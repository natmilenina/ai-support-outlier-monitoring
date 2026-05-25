from __future__ import annotations

NON_RESPONSE_CATEGORY_COLUMNS = {
    "nr_small_talk": "small_talk",
    "nr_statement": "statement",
    "nr_unclear_request": "unclear_request",
    "nr_out_of_scope": "out_of_scope",
    "nr_noise_or_spam": "noise_or_spam",
    "nr_user_to_user": "user_to_user",
    "nr_missing_context": "missing_context",
    "nr_knowledge_not_found": "knowledge_not_found",
    "nr_malicious": "malicious",
    "nr_other": "other",
}

NON_RESPONSE_CATEGORIES = list(NON_RESPONSE_CATEGORY_COLUMNS.values())

EXPORT_ELIGIBLE_CATEGORIES = {
    "statement",
    "unclear_request",
    "out_of_scope",
    "missing_context",
    "knowledge_not_found",
    "malicious",
    "other",
}

TRACK_ONLY_CATEGORIES = {
    "small_talk",
    "noise_or_spam",
    "user_to_user",
}

DAILY_TOTALS_COLUMNS = [
    "message_date",
    "total_messages",
    "answered_messages",
    "non_response_total",
    *NON_RESPONSE_CATEGORY_COLUMNS.keys(),
]

DAILY_BY_CHANNEL_COLUMNS = [
    "message_date",
    "channel_name",
    "channel_id",
    "total_messages",
    "answered_messages",
    "non_response_total",
    *NON_RESPONSE_CATEGORY_COLUMNS.keys(),
]

RAW_MESSAGES_COLUMNS = [
    "timestamp_utc",
    "request_date",
    "channel_name",
    "channel_id",
    "demo_message_link",
    "message_id",
    "user_request",
    "language",
    "bot_response",
    "non_response_category",
    "session_info",
]

import pandas as pd


CATEGORY_REVIEW_LIBRARY = {
    "statement": {
        "classification_quality": "mixed",
        "main_themes": "Support issues phrased as statements",
        "recommended_action": "Review whether statement-like messages with clear support intent should receive bot answers.",
    },
    "unclear_request": {
        "classification_quality": "mixed",
        "main_themes": "Short or vague support requests",
        "recommended_action": "Improve clarification behavior for vague messages instead of immediately treating them as non-responses.",
    },
    "out_of_scope": {
        "classification_quality": "mostly_valid",
        "main_themes": "Requests outside supported product scope",
        "recommended_action": "Check whether repeated out-of-scope topics reveal emerging user needs.",
    },
    "missing_context": {
        "classification_quality": "mostly_valid",
        "main_themes": "Follow-up messages without enough context",
        "recommended_action": "Review whether conversation context handling can be improved.",
    },
    "knowledge_not_found": {
        "classification_quality": "likely_problem",
        "main_themes": "Possible documentation or knowledge coverage gaps",
        "recommended_action": "Review examples and consider adding documentation or prompt coverage.",
    },
    "malicious": {
        "classification_quality": "mixed",
        "main_themes": "Potential prompt-injection or policy-risk messages",
        "recommended_action": "Review false positives carefully so valid requests are not blocked.",
    },
    "other": {
        "classification_quality": "likely_problem",
        "main_themes": "Fallback category absorbing too many messages",
        "recommended_action": "Review taxonomy and ensure 'other' is used only as a final fallback.",
    },
}


def build_pregenerated_review(
    raw_messages_df: pd.DataFrame,
    message_date,
    category: str,
) -> dict:
    guidance = CATEGORY_REVIEW_LIBRARY.get(
        category,
        {
            "classification_quality": "mostly_valid",
            "main_themes": "General non-response pattern",
            "recommended_action": "Review examples to confirm whether this category behavior is expected.",
        },
    )

    examples = (
        raw_messages_df["user_request"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    possible_misclassifications = examples[:3]
    should_bot_have_answered = []

    if category in {"statement", "unclear_request", "knowledge_not_found", "other"}:
        should_bot_have_answered = examples[:2]

    return {
        "message_date": str(message_date),
        "category": category,
        "classification_quality": guidance["classification_quality"],
        "main_themes": guidance["main_themes"],
        "possible_misclassifications": " | ".join(possible_misclassifications),
        "should_bot_have_answered": " | ".join(should_bot_have_answered),
        "recommended_action": guidance["recommended_action"],
        "slack_summary": (
            f"{category} spiked on {message_date}. "
            f"Theme: {guidance['main_themes']}. "
            f"Recommended action: {guidance['recommended_action']}"
        ),
        "sample_size": len(raw_messages_df),
        "review_source": "pre_generated_demo_review",
    }

import json
import os

import pandas as pd
from dotenv import load_dotenv
from google import genai
import streamlit as st


def build_gemini_review_prompt(
    raw_messages_df: pd.DataFrame,
    category: str,
    message_date,
    max_messages: int = 30,
) -> str:
    sample_df = raw_messages_df[
        [
            "timestamp_utc",
            "channel_name",
            "demo_message_link",
            "user_request",
            "non_response_category",
        ]
    ].head(max_messages).copy()

    for column in sample_df.columns:
        sample_df[column] = sample_df[column].astype(str)

    messages = sample_df.to_dict(orient="records")

    return f"""
You are reviewing non-response classifications for an AI support assistant.

The assistant intentionally did not answer these messages because they were classified as:

Category: {category}
Date: {message_date}

Analyze only this date/category group.

Your task:
1. Identify the main themes in the messages.
2. Evaluate whether the assigned non-response category appears correct.
3. Identify examples that may require human review.
4. Determine whether the assistant may have been expected to answer any messages.
5. Suggest whether this points to:
   - classification issue
   - missing knowledge/docs
   - community moderation/spam issue
   - expected behavior
   - taxonomy issue
6. Provide a short summary suitable for a Slack alert.

Return valid JSON only. Do not wrap it in markdown.

Use this exact structure:

{{
  "classification_quality": "valid | mostly_valid | mixed | likely_problem",
  "main_themes": ["theme 1", "theme 2", "theme 3"],
  "possible_misclassifications": [
    {{
      "message": "short quoted or paraphrased user message",
      "why_it_may_be_misclassified": "reason"
    }}
  ],
  "should_bot_have_answered": [
    {{
      "message": "short quoted or paraphrased user message",
      "reason": "why the assistant may have been expected to answer"
    }}
  ],
  "recommended_action": "concise recommendation",
  "slack_summary": "short summary for Slack"
}}

Messages:
{json.dumps(messages, ensure_ascii=False, indent=2)}
"""


def parse_llm_json_response(response_text: str) -> dict:
    cleaned = response_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def run_gemini_review(
    raw_messages_df: pd.DataFrame,
    category: str,
    message_date,
    max_messages: int = 30,
) -> dict:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") or st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

    client = genai.Client(api_key=api_key)

    prompt = build_gemini_review_prompt(
        raw_messages_df=raw_messages_df,
        category=category,
        message_date=message_date,
        max_messages=max_messages,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    response_text = response.text or ""

    try:
        parsed = parse_llm_json_response(response_text)
    except json.JSONDecodeError:
        parsed = {
            "classification_quality": "parse_error",
            "main_themes": [],
            "possible_misclassifications": [],
            "should_bot_have_answered": [],
            "recommended_action": "Could not parse Gemini response as JSON.",
            "slack_summary": response_text,
        }

    parsed["model_name"] = model_name
    parsed["sample_size"] = min(len(raw_messages_df), max_messages)
    parsed["review_source"] = "live_gemini"

    return parsed

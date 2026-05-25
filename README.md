# AI Support Outlier Monitoring

Operational analytics workflow for monitoring and investigating AI support assistant intent categorization failures and blind spots.

## Overview

This project simulates a production-style monitoring workflow for AI support assistants operating in high-volume online communities.

The workflow focuses on detecting abnormal spikes in non-response intent categories — situations where the assistant intentionally decides not to answer user messages.

Monitoring these categories helps identify:
- potential classification failures
- missing documentation
- fallback overuse
- taxonomy degradation
- moderation or spam-driven traffic anomalies

The project demonstrates:
- statistical outlier detection
- rolling historical baselines
- category-specific behavioral baselines
- share-average and percentile-based anomaly detection
- minimum-volume safeguards for low-traffic categories
- raw-message investigation workflows
- LLM-assisted operational review
- Slack-style escalation summaries
- Streamlit analytics interface design

All datasets included in this repository are synthetic.

---

## Demo workflow

The application demonstrates a multi-step operational analytics workflow:

1. Select a detection window
2. Compare traffic against historical baselines
3. Detect abnormal category spikes
4. Review raw message samples
5. Run optional Gemini-powered investigation
6. Generate operational Slack-style summaries

---

## Example investigation questions

The workflow helps investigate situations such as:

- Are eligible user questions being ignored?
- Is documentation coverage missing for a new feature?
- Is a fallback category absorbing too much traffic?
- Did moderation or spam events affect support quality?
- Has intent classification quality degraded?

---

## Tech stack

- Python
- Pandas
- Streamlit
- Gemini API
- Statistical anomaly detection

---

## Project structure

```text
.
├── app.py
├── data/
├── src/
├── generate_data.py
├── generate_reviews.py
├── requirements.txt
└── test_outliers.py
```

---

## Local setup

Clone the repository:

```bash
git clone https://github.com/natmilenina/ai-support-outlier-monitoring.git
cd ai-support-outlier-monitoring
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment variables:

```bash
cp .env.example .env
```

Add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Run the application:

```bash
streamlit run app.py
```

---

## Notes

- All datasets are synthetic and generated for demonstration purposes.
- The live Gemini investigation step is optional.
- The production inspiration for this workflow came from operational monitoring challenges in large-scale AI support environments.

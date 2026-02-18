---

# Gmail Janitor - AI-Powered Email Cleanup (v2 Spec)

A Python tool that uses **Gemini 2.5 Flash (Vertex AI)** to triage Gmail emails by **category + importance/junk scoring**, then applies a **risk-aware deletion policy** (trash / label for review / keep) that improves over time through **user follow-up questions** (active learning + preference memory).

---

## Core Goals

1. **Reduce inbox clutter safely** (never “hard delete”).
2. **Personalize decisions to the user** via follow-up feedback and remembered preferences.
3. **Lower cost and increase speed** via rule-based prefiltering + caching, only calling Gemini when needed.
4. **Explainable, reversible actions** (review label + undo + audit log).

---

## High-Level Pipeline

### Stage 0 — User Context + Preference Loading

* Load per-account preferences and feedback:

  * `user_preferences.json`
  * `feedback_log.json`
  * `sender_stats.json`
  * `cache_classifications.json`

### Stage 1 — Keyword Search (Optional)

* User provides keywords (like your current flow), but now keywords are just the *candidate set*.
* Supports additional search modes (optional):

  * `--recent 30d`
  * `--label Promotions`
  * `--unread-only`
  * `--from-domain linkedin.com`

### Stage 2 — Rule-Based Pre-Filtering (Deterministic)

Before calling Gemini:

* Hard keep rules:

  * Whitelisted senders/domains
  * Receipts (e.g., “receipt”, “invoice”) unless user says otherwise
* Hard trash rules (optional, conservative):

  * Known spam domains
  * Explicit user “always trash” patterns
* Any “hard decision” emails are **not sent to Gemini**.

### Stage 3 — Gemini Scoring + Categorization (Only for Remaining Emails)

Gemini returns structured output (JSON schema) with:

* `category`: `job_alert | marketing | receipt | financial | social | personal | system | unknown`
* `importance_score` (0–1)
* `junk_score` (0–1)
* `risk_of_wrong_deletion` (0–1)
* `confidence` (0–1)
* `reasoning` (short, non-sensitive)

### Stage 4 — Decision Policy (User-Controlled Thresholds)

The system converts Gemini signals into actions:

* **KEEP** if `importance_score >= KEEP_THRESHOLD`
* **TRASH** if `junk_score >= TRASH_THRESHOLD` and `risk_of_wrong_deletion <= RISK_MAX`
* **REVIEW** (safe default) otherwise → apply a label (not trash)

**Modes (important):**

* **Conservative**: default to Review if unsure
* **Balanced**: trash high-confidence junk
* **Aggressive**: trash medium-confidence junk (still not hard delete)

### Stage 5 — Active Learning Follow-Up Questions (Mandatory v2 Feature)

After results display, the tool asks follow-ups ONLY for **high-value emails**:

* emails near thresholds (uncertain)
* emails from new senders/domains
* categories the user frequently overrides

User answers:

* “Correct?” (Y/N)
* If incorrect: Keep/Trash/Review
* Optional reason: “job-related”, “marketing spam”, “receipt I need”, etc.

These answers update:

* preference weights (probabilistic memory)
* prompt rules
* sender reputation stats
* future thresholds (optional)

---

## Data & Memory (Per Account)

All stored locally per Gmail account:

### `user_preferences.json`

Stores:

* decision mode (Conservative/Balanced/Aggressive)
* keep/trash thresholds
* category rules (e.g., “keep job_alert always”)
* sender/domain weights

Example structure:

```json
{
  "account": "denissoulima@gmail.com",
  "mode": "conservative",
  "thresholds": {
    "keep": 0.75,
    "trash": 0.85,
    "risk_max": 0.2
  },
  "category_rules": {
    "job_alert": "keep",
    "receipt": "keep",
    "marketing": "review"
  },
  "sender_weights": [
    {
      "pattern": "linkedin.com",
      "keep_weight": 0.65,
      "trash_weight": 0.35,
      "num_feedbacks": 12
    }
  ]
}
```

### `sender_stats.json`

Tracks:

* frequency of emails per sender/domain
* override rate (how often AI was wrong)
* average importance/junk score by sender

### `feedback_log.json`

Stores user corrections (with timestamps, category, and what changed).

### `cache_classifications.json`

Caches message-id → classification JSON so reruns don’t reclassify identical emails.

---

## Safety & Reversibility

### Soft Delete (Default)

Instead of trashing immediately, the default behavior is:

* Apply label: `GmailJanitor/Review`
* Optionally archive
* Auto-trash after `N days` (configurable), unless user disables

### Undo Support (Recommended)

Maintain an `actions_log.json` containing message IDs moved to trash or labeled.

CLI:

* `--undo-last` restores previous batch (remove label / move out of trash if possible)

### Audit Log (Always-On)

Log:

* query used
* number analyzed
* counts: keep / review / trash
* estimated cost (optional)
* timestamped actions

---

## Gemini Prompting (Adaptive System Prompt Engine)

Instead of a static prompt, the system generates a **dynamic prompt** per run:

Inputs:

* user preferences (thresholds, category rules)
* sender stats (frequent senders, likely junk)
* recent user feedback (last 20 corrections)
* current search context (keywords, date range, unread-only)

This prompt is used as `system_instruction`.

---

## Interactive Usage Flow (Updated)

### Interactive Script

```bash
.venv\Scripts\python.exe main.py
```

Updated flow:

1. Authenticate with Gmail
2. Choose search mode:

   * keywords (default)
   * recent / unread / label
3. Ask for “importance preferences” (first run or if user chooses):

   * “Are job alerts important?”
   * “Do you want receipts always kept?”
   * “Do you want newsletters reviewed or trashed?”
4. Search Gmail candidate set
5. Rule-based prefilter decisions (keep/trash) where confident
6. Classify remaining emails with Gemini (batches)
7. Apply decision policy → Keep / Review label / Trash
8. Show summary + ask follow-up questions (active learning)
9. Save updated preferences + stats + cache
10. Apply actions (confirm required unless running in `--auto` mode)

---

## CLI Options (Recommended Additions)

* `--mode conservative|balanced|aggressive`
* `--keep-threshold 0.75`
* `--trash-threshold 0.85`
* `--risk-max 0.2`
* `--review-label GmailJanitor/Review`
* `--no-trash` (label-only)
* `--auto` (no prompts; uses saved preferences)
* `--undo-last`
* `--recent 30d`
* `--unread-only`
* `--max N`
* `--dry-run`

---

## Vertex AI / Gemini Notes

* Supports structured JSON output schema (Pydantic JSON schema).
* Uses batching to reduce calls.
* Rate limiting + retry on 429 remains.

**Cost optimization:**

* rule-based prefilter reduces Gemini calls
* caching prevents re-classification across runs

---

## Jupyter Notebook (Optional Enhancements)

Add analytics:

* top senders by volume
* category distribution over time
* estimated clutter reduction
* override accuracy trend

---

## Setup Guide (Same + Clarifications)

### 1. Google Cloud Project

(same as your current instructions)

### 2. Enable Gmail API

(same)

### 3. Configure OAuth Consent Screen

(same)

### 4. Create OAuth Credentials

(same)

### 5. Vertex AI Setup (If Using Vertex Mode)

* Enable **Vertex AI API**
* Ensure region is set (e.g., `us-central1`)
* Use ADC credentials (your existing OAuth flow covers Gmail; Vertex uses gcloud creds or service account depending on your setup)

> If you keep using `genai.Client(vertexai=True, project=..., location=...)`, document that the user must be authenticated to GCP locally (or provide service account flow).

### 6. Install Dependencies

(same)

---

## Notes (Updated)

* **Testing mode**: OAuth tokens may expire; script handles refresh and re-auth.
* **Safety**: Default behavior is **Review Label + optional delayed trash**.
* **Trust**: The tool asks follow-up questions for uncertain cases and remembers preferences.
* **Privacy**: Only minimal email metadata (sender, subject, snippet) should be sent to Gemini; optionally allow `--no-snippet` mode for stricter privacy.

---

## Recommended Next Implementation Order

1. **Decision scoring schema + thresholds + modes**
2. **Review label + audit log + undo**
3. **Active learning follow-up questions (uncertain emails only)**
4. **Preference memory + sender stats**
5. **Rule-based prefilter + caching**
6. (Optional) embeddings + similarity retrieval memory

---

If you want, I can also rewrite your existing `prompts.yml` into:

* a dynamic template builder (with injected preferences + recent feedback)
* a JSON schema that includes the new fields (category, scores, risk)

…and sketch the minimal code diffs to implement **Review mode + active learning** first.

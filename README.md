# Gmail Janitor — AI-Powered Email Cleanup

> **v2.3** | Uses **Gemini 2.5 Flash** to classify emails by category + importance/junk scoring, then applies a risk-aware deletion policy that improves over time through active learning and preference memory.

![Gmail Janitor — Signed In](Screenshots%20for%20ReadMe/Screenshot2%20Gmail%20Janitor.png)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [High-Level Pipeline](#high-level-pipeline)
  - [How the LLM Classifier Works](#how-the-llm-classifier-works)
  - [Application Diagram](#application-diagram)
- [Tech Stack](#tech-stack)
- [Directory Overview](#directory-overview)
- [Screenshots](#screenshots)
- [Prerequisites](#prerequisites)
- [Setup Guide (Step by Step)](#setup-guide-step-by-step)
  - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
  - [Step 2 — Create a Google Cloud Project](#step-2--create-a-google-cloud-project)
  - [Step 3 — Enable the Gmail API](#step-3--enable-the-gmail-api)
  - [Step 4 — Configure OAuth Consent Screen](#step-4--configure-oauth-consent-screen)
  - [Step 5 — Create OAuth Credentials and Download credentials.json](#step-5--create-oauth-credentials-and-download-credentialsjson)
  - [Step 6 — Get a Gemini API Key](#step-6--get-a-gemini-api-key)
  - [Step 7 — Configure Environment Variables](#step-7--configure-environment-variables)
  - [Step 8 — Install Dependencies](#step-8--install-dependencies)
  - [Step 9 — Run the App](#step-9--run-the-app)
- [How to Use the App](#how-to-use-the-app)
- [CLI Usage (Alternative)](#cli-usage-alternative)
- [Safety and Reversibility](#safety-and-reversibility)
- [License](#license)

---

## Overview

Gmail Janitor is a full-stack AI-powered email cleanup tool that connects to your Gmail account via the Gmail API and uses Google's **Gemini 2.5 Flash** LLM to intelligently classify, score, and triage your emails. It features a modern **Streamlit** web UI with:

- **Multi-mode search** — Keywords, Recent, Label, From Domain
- **AI classification** — Emails scored on importance, junk probability, and deletion risk
- **Risk-aware decision policy** — Conservative / Balanced / Aggressive modes
- **Quarantine system** — Review labeled emails before permanent action
- **Natural language commands** — Type commands like "Trash all marketing emails" and the AI parses them into structured action plans
- **Active learning** — The system asks follow-up questions for uncertain classifications and remembers your preferences
- **Undo support** — Every action is logged and reversible

---

## Architecture

### High-Level Pipeline

The app processes emails through a 5-stage pipeline:

```
+------------------+     +------------------+     +-------------------+
|  Stage 0         |     |  Stage 1         |     |  Stage 2          |
|  Load User       |---->|  Search Gmail    |---->|  Rule-Based       |
|  Preferences &   |     |  (Keywords,      |     |  Pre-Filtering    |
|  Feedback Data   |     |   Recent, Label, |     |  (Whitelist,      |
|                  |     |   Domain)        |     |   Blacklist,      |
+------------------+     +------------------+     |   Receipts)       |
                                                  +--------+----------+
                                                           |
                                    +----------------------+
                                    |
                         Hard Keep / Hard Trash        Needs AI
                         (skip Gemini)                     |
                                                           v
                                                  +-------------------+
                                                  |  Stage 3          |
                                                  |  Gemini 2.5 Flash |
                                                  |  LLM Classifier   |
                                                  |  (Batch scoring   |
                                                  |   with dynamic    |
                                                  |   system prompt)  |
                                                  +--------+----------+
                                                           |
                                                           v
                                                  +-------------------+
                                                  |  Stage 4          |
                                                  |  Decision Policy  |
                                                  |  (Threshold-based |
                                                  |   Keep/Review/    |
                                                  |   Trash)          |
                                                  +--------+----------+
                                                           |
                                                           v
                                                  +-------------------+
                                                  |  Stage 5          |
                                                  |  Active Learning  |
                                                  |  Follow-Up &      |
                                                  |  Execute Actions  |
                                                  +-------------------+
```

### How the LLM Classifier Works

The Gemini 2.5 Flash model is called via the `google-genai` SDK with **structured JSON output** (Pydantic schema enforcement). For each email, the model returns:

| Field                    | Type    | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `category`               | string  | `job_alert`, `marketing`, `receipt`, `financial`, `social`, `personal`, `system`, `unknown` |
| `importance_score`       | 0.0–1.0 | How important the email is to the user           |
| `junk_score`             | 0.0–1.0 | Likelihood the email is junk/marketing           |
| `risk_of_wrong_deletion` | 0.0–1.0 | Risk that deleting this email would be a mistake |
| `confidence`             | 0.0–1.0 | Model's confidence in the classification         |
| `reasoning`              | string  | Brief explanation of the classification          |

**Key design decisions:**

1. **Dynamic system prompt** — The prompt is built per-run by injecting user preferences, sender reputation stats, recent feedback corrections, and the current search context. This means the AI learns from your past corrections.

2. **Batch processing** — Emails are sent to Gemini in batches of 20 with rate limiting and retry logic for 429 errors.

3. **Caching** — Classifications are cached by message ID so re-runs don't re-classify the same emails.

4. **Pre-filtering** — Whitelisted senders, receipts, and blacklisted domains are handled by deterministic rules before calling Gemini, reducing API cost.

5. **Decision policy** — The final Keep/Review/Trash decision uses configurable thresholds that vary by mode:

   | Mode         | Trash Threshold | Risk Max |
   |--------------|-----------------|----------|
   | Conservative | 0.85            | 0.20     |
   | Balanced     | 0.80            | 0.25     |
   | Aggressive   | 0.65            | 0.35     |

6. **Natural language command planner** — A separate Gemini call (`planner_service.py`) parses free-text commands like "Trash all LinkedIn emails" into structured `ActionPlan` objects with search parameters, action types, and preference updates.

### Application Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌───────────┐  │
│  │Run       │  │Quarantine │  │Rules &       │  │History &  │  │
│  │Cleanup   │  │           │  │Preferences   │  │Stats      │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘  └─────┬─────┘  │
│       │              │               │               │         │
│  ┌────┴──────────────┴───────────────┴───────────────┴────┐    │
│  │                    Session State                        │    │
│  └────────────────────────┬───────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            v               v               v
   ┌────────────┐  ┌────────────┐  ┌─────────────────┐
   │  main.py   │  │ planner_   │  │  prompts.yml    │
   │  Backend   │  │ service.py │  │  Prompt          │
   │  Engine    │  │ NL Command │  │  Templates       │
   │            │  │ Parser     │  │                  │
   └──┬────┬───┘  └──────┬─────┘  └─────────────────┘
      │    │              │
      │    │              v
      │    │     ┌─────────────────┐
      │    │     │  Gemini 2.5     │
      │    └────>│  Flash API      │
      │          │  (google-genai) │
      v          └─────────────────┘
┌──────────┐
│ Gmail    │     ┌─────────────────────────────┐
│ API      │     │  data/<account>/            │
│ (OAuth2) │     │  ├── user_preferences.json  │
└──────────┘     │  ├── feedback_log.json      │
                 │  ├── sender_stats.json       │
                 │  ├── cache_classifications   │
                 │  └── actions_log.json        │
                 └─────────────────────────────┘
```

---

## Tech Stack

| Component            | Technology                                                      |
|----------------------|-----------------------------------------------------------------|
| **Frontend**         | [Streamlit](https://streamlit.io/) (Python web framework)      |
| **LLM**             | [Gemini 2.5 Flash](https://ai.google.dev/) via `google-genai`  |
| **Email API**        | [Gmail API v1](https://developers.google.com/gmail/api) (OAuth 2.0) |
| **Data Validation**  | [Pydantic v2](https://docs.pydantic.dev/) (structured LLM output) |
| **Prompt Management**| YAML templates (`prompts.yml`)                                  |
| **NL Command Parser**| Custom Gemini-based planner (`planner_service.py`)              |
| **Environment**      | `python-dotenv`, `.env` file                                    |
| **Notebook**         | Jupyter + Plotly + Pandas (analytics / visualization)           |
| **Language**         | Python 3.11+                                                    |

---

## Directory Overview

```
Gmail Janitor/
├── app.py                    # Streamlit web UI (frontend)
├── main.py                   # Backend engine (auth, search, classify, decide, execute)
├── planner_service.py        # Natural language command parser (Gemini-powered)
├── prompts.yml               # LLM prompt templates (system prompt, scoring guidelines)
├── requirements.txt          # Python dependencies
├── credentials.json          # Google OAuth credentials (YOU must download this — see setup)
├── .env                      # API keys and config (not committed)
├── .gitignore                # Ignores .env, credentials.json, tokens/, data/
│
├── tokens/                   # OAuth tokens per account (auto-generated)
│   └── token_<email>.json
│
├── data/                     # Per-account persistent data (auto-generated)
│   └── <email>/
│       ├── user_preferences.json       # Decision mode, thresholds, category rules
│       ├── feedback_log.json           # User corrections for active learning
│       ├── sender_stats.json           # Sender reputation tracking
│       ├── cache_classifications.json  # Cached Gemini classifications
│       └── actions_log.json            # Audit log of all actions (for undo)
│
├── Screenshots for ReadMe/   # UI screenshots
│   ├── Screenshot1 For GenAI Submission.png
│   └── Screenshot2 Gmail Janitor.png
│
├── janitor_notebook.ipynb    # Jupyter notebook for analytics and visualization
├── FRONTENDSPEC.md           # Frontend specification document
└── README.md                 # This file
```

---

## Screenshots

### Login Screen
Select your Google account and connect via OAuth:

![Login Screen](Screenshots%20for%20ReadMe/Screenshot1%20For%20GenAI%20Submission.png)

### Main Dashboard (Authenticated)
Search, classify, and clean up emails with AI:

![Main Dashboard](Screenshots%20for%20ReadMe/Screenshot2%20Gmail%20Janitor.png)

---

## Prerequisites

- **Python 3.11+**
- **A Google account** with Gmail
- **A Google Cloud Platform (GCP) project** with the Gmail API enabled
- **A Gemini API key** (free from Google AI Studio)

---

## Setup Guide (Step by Step)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/dehiska/GmailJanitor.git
cd GmailJanitor
```

### Step 2 — Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** > **New Project**
3. Name it (e.g., `Gmail Janitor`) and click **Create**
4. Make sure the new project is selected in the top dropdown

### Step 3 — Enable the Gmail API

1. In the GCP Console, go to **APIs & Services > Library**
2. Search for **"Gmail API"**
3. Click on it and press **Enable**

### Step 4 — Configure OAuth Consent Screen

1. Go to **APIs & Services > OAuth consent screen**
2. Select **External** user type and click **Create**
3. Fill in the required fields:
   - **App name**: `Gmail Janitor`
   - **User support email**: your email
   - **Developer contact**: your email
4. Click **Save and Continue**
5. On the **Scopes** page, click **Add or remove scopes** and add:
   - `https://www.googleapis.com/auth/gmail.modify`
6. Click **Save and Continue**
7. On the **Test users** page, add your Gmail address as a test user
8. Click **Save and Continue** then **Back to Dashboard**

### Step 5 — Create OAuth Credentials and Download credentials.json

> **This is a crucial step.** You must download the `credentials.json` file from GCP and place it in the project root.

1. Go to **APIs & Services > Credentials**
2. Click **+ CREATE CREDENTIALS > OAuth client ID**
3. Set **Application type** to **Desktop app**
4. Name it (e.g., `Gmail Janitor Desktop`)
5. Click **Create**
6. In the popup, click **DOWNLOAD JSON**
7. **Rename** the downloaded file to `credentials.json`
8. **Move** it to the root of the project directory:
   ```
   Gmail Janitor/
   └── credentials.json   <-- place it here
   ```

> **Warning:** Never commit `credentials.json` to version control. It is already in `.gitignore`.

### Step 6 — Get a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy the key — you will need it in the next step

> **Alternative:** If you prefer Vertex AI mode, enable the **Vertex AI API** in your GCP project and use your GCP project ID instead of an API key.

### Step 7 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Option 1: API Key mode (simplest)
GEMINI_API_KEY=your-gemini-api-key-here

# Option 2: Vertex AI mode (uses GCP credits)
# GCP_PROJECT=your-gcp-project-id
# GCP_LOCATION=us-central1
```

> **Warning:** Never commit your `.env` file. It is already in `.gitignore`.

### Step 8 — Install Dependencies

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Step 9 — Run the App

```bash
# Start the Streamlit web UI
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

On first run, you will be prompted to authenticate with Google. A browser window will open for the OAuth consent flow. After granting permissions, your token is saved locally in the `tokens/` directory.

---

## How to Use the App

### 1. Sign In
- In the sidebar, select your account (if previously authenticated) and click **Connect**
- Or click **Sign in to new account** to authenticate via Google OAuth

### 2. Configure Settings (Sidebar)
- **Mode**: Choose `conservative`, `balanced`, or `aggressive`
  - Conservative = safest, defaults to Review for uncertain emails
  - Aggressive = trashes more, for experienced users
- **Thresholds**: Expand to fine-tune keep/trash/risk thresholds
- **Quarantine Label**: Set the Gmail label used for emails flagged for review

### 3. Search for Emails (Run Cleanup Tab)
- Choose a **Search mode**:
  - **Keywords** — Comma-separated terms (e.g., `LinkedIn, Coursera, Best Buy`)
  - **Recent** — Process emails from the last N days
  - **Label** — Search within a Gmail label
  - **From domain** — Filter by sender domain (e.g., `linkedin.com`)
- Set **Max emails to process**
- Toggle **Unread only** if desired

### 4. Preview Classification
- Click **Preview Classification** to run the AI pipeline
- The app will:
  1. Search Gmail for matching emails
  2. Pre-filter using deterministic rules (whitelist/blacklist/receipts)
  3. Send remaining emails to Gemini for AI classification
  4. Apply the decision policy (Keep / Review / Trash)
  5. Display results in an interactive table

### 5. Review Results
- Review the classification results showing category, scores, and recommended action
- Override individual decisions if needed

### 6. Execute Actions
- Click **Execute Actions** to apply the decisions:
  - **Keep**: No action taken
  - **Review**: Email gets labeled with your quarantine label
  - **Trash**: Email is moved to Gmail Trash (soft delete, recoverable for 30 days)

### 7. Natural Language Commands (Sidebar)
- Use the **Command** section to type natural language instructions:
  - `"Trash all marketing emails"`
  - `"Keep all job alerts"`
  - `"Undo my last cleanup"`
- The AI parses your command into a structured action plan

### 8. Quarantine Tab
- Review emails that were labeled for review
- Approve or override the AI's suggestion

### 9. Rules & Preferences Tab
- Manage whitelist/blacklist domains
- Set category rules (always keep receipts, always trash marketing, etc.)
- Configure always-trash patterns

### 10. History & Stats Tab
- View past cleanup runs with timestamps and action counts
- Undo previous actions if needed

---

## CLI Usage (Alternative)

The backend can also be run directly from the command line:

```bash
python main.py
```

### CLI Options

```
--mode conservative|balanced|aggressive   Decision mode
--keep-threshold 0.75                     Importance score threshold for keeping
--trash-threshold 0.85                    Junk score threshold for trashing
--risk-max 0.2                            Max risk for trashing
--review-label GmailJanitor/Review        Gmail label for review
--no-trash                                Label-only mode (never trash)
--auto                                    No interactive prompts
--undo-last                               Undo the most recent batch
--recent 30d                              Search recent emails
--unread-only                             Only unread emails
--from-domain linkedin.com                Filter by domain
--max 50                                  Max emails to process
--dry-run                                 Show results without executing
--keywords "LinkedIn,Coursera"            Comma-separated keywords
```

---

## Safety and Reversibility

- **No hard deletes** — Emails are moved to Gmail Trash (auto-deleted by Gmail after 30 days) or labeled for review
- **Undo support** — Every action is logged in `actions_log.json` with message IDs; use undo to restore
- **Conservative defaults** — The system defaults to "Review" when uncertain
- **Audit trail** — Full history of all actions with timestamps
- **Pre-filtering** — Receipts, financial emails, and whitelisted senders are protected by deterministic rules before the AI even sees them

---

## License

This project was built as part of a Generative AI course. For educational use.

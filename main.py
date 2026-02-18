"""
Gmail Janitor v2 - AI-Powered Email Cleanup
Uses Gemini 2.5 Flash to classify emails by category + importance/junk scoring,
then applies a risk-aware deletion policy (trash / label for review / keep)
that improves over time through user follow-up questions (active learning + preference memory).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tabulate import tabulate
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google import genai

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
PROJECT_DIR = Path(__file__).parent
CREDENTIALS_FILE = PROJECT_DIR / "credentials.json"
TOKENS_DIR = PROJECT_DIR / "tokens"
DATA_DIR = PROJECT_DIR / "data"
PROMPTS_FILE = PROJECT_DIR / "prompts.yml"
MAX_RESULTS_PER_PAGE = 100
GEMINI_MODEL = "gemini-2.5-flash"
RATE_LIMIT_DELAY = 1.5
BATCH_SIZE = 20

RECEIPT_KEYWORDS = ["receipt", "invoice", "order confirmation", "payment confirmation",
                    "shipping confirmation", "delivery confirmation", "purchase"]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class EnhancedEmailClassification(BaseModel):
    email_index: int = Field(
        description="The index number of the email being classified"
    )
    category: Literal[
        "job_alert", "marketing", "receipt", "financial",
        "social", "personal", "system", "unknown"
    ] = Field(
        description="The category of the email"
    )
    importance_score: float = Field(
        ge=0.0, le=1.0,
        description="How important is this email to the user (0=not, 1=critical)"
    )
    junk_score: float = Field(
        ge=0.0, le=1.0,
        description="How likely this email is junk/marketing (0=not junk, 1=definitely junk)"
    )
    risk_of_wrong_deletion: float = Field(
        ge=0.0, le=1.0,
        description="Risk that deleting this would be a mistake (0=safe to delete, 1=never delete)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in this classification"
    )
    reasoning: str = Field(
        description="Brief reasoning for the classification"
    )


class BatchClassification(BaseModel):
    classifications: list[EnhancedEmailClassification] = Field(
        description="List of classifications, one per email in the batch"
    )


class SenderWeight(BaseModel):
    pattern: str
    keep_weight: float = 0.5
    trash_weight: float = 0.5
    num_feedbacks: int = 0


class UserPreferences(BaseModel):
    account: str = ""
    mode: Literal["conservative", "balanced", "aggressive"] = "conservative"
    thresholds: dict = Field(default_factory=lambda: {
        "keep": 0.75,
        "trash": 0.85,
        "risk_max": 0.2,
    })
    category_rules: dict = Field(default_factory=lambda: {
        "job_alert": "keep",
        "receipt": "keep",
        "financial": "keep",
        "marketing": "review",
        "social": "review",
        "personal": "keep",
        "system": "keep",
        "unknown": "review",
    })
    sender_weights: list[SenderWeight] = Field(default_factory=list)
    whitelist_domains: list[str] = Field(default_factory=list)
    whitelist_senders: list[str] = Field(default_factory=list)
    blacklist_domains: list[str] = Field(default_factory=list)
    always_trash_patterns: list[str] = Field(default_factory=list)


class FeedbackEntry(BaseModel):
    timestamp: str
    message_id: str
    sender: str
    subject: str
    original_decision: str
    corrected_decision: str
    category: str
    reason: str = ""


class SenderStat(BaseModel):
    sender_or_domain: str
    total_emails: int = 0
    times_trashed: int = 0
    times_kept: int = 0
    times_reviewed: int = 0
    override_count: int = 0
    avg_importance: float = 0.0
    avg_junk: float = 0.0


class ActionLogEntry(BaseModel):
    timestamp: str
    action: str
    message_ids: list[str] = Field(default_factory=list)
    label_ids: list[str] = Field(default_factory=list)
    query_used: str = ""
    counts: dict = Field(default_factory=dict)
    # v2.2: track attempted vs succeeded vs failed
    attempted_ids: list[str] = Field(default_factory=list)
    succeeded_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gmail Janitor v2 - AI-Powered Email Cleanup"
    )
    parser.add_argument("--mode", choices=["conservative", "balanced", "aggressive"],
                        help="Decision mode (default: from saved prefs or conservative)")
    parser.add_argument("--keep-threshold", type=float,
                        help="Importance score threshold for keeping (default: 0.75)")
    parser.add_argument("--trash-threshold", type=float,
                        help="Junk score threshold for trashing (default: 0.85)")
    parser.add_argument("--risk-max", type=float,
                        help="Max risk_of_wrong_deletion for trashing (default: 0.2)")
    parser.add_argument("--review-label", default="GmailJanitor/Review",
                        help="Gmail label for review emails (default: GmailJanitor/Review)")
    parser.add_argument("--no-trash", action="store_true",
                        help="Label-only mode: never trash, only label for review")
    parser.add_argument("--auto", action="store_true",
                        help="No interactive prompts; use saved preferences")
    parser.add_argument("--undo-last", action="store_true",
                        help="Undo the most recent batch of actions")
    parser.add_argument("--recent", type=str,
                        help="Search recent emails (e.g., '30d' for last 30 days)")
    parser.add_argument("--unread-only", action="store_true",
                        help="Only search unread emails")
    parser.add_argument("--label", type=str,
                        help="Search within a specific Gmail label")
    parser.add_argument("--from-domain", type=str,
                        help="Search emails from a specific domain")
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of emails to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show results without executing any actions")
    parser.add_argument("--keywords", type=str,
                        help="Comma-separated keywords (alternative to interactive prompt)")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts() -> dict:
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Gemini client initialization (portable)
# ---------------------------------------------------------------------------

def _read_project_from_credentials() -> str | None:
    """Try to extract project_id from credentials.json."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        with open(CREDENTIALS_FILE, "r") as f:
            creds_data = json.load(f)
        for key in ("installed", "web"):
            if key in creds_data and "project_id" in creds_data[key]:
                return creds_data[key]["project_id"]
    except Exception:
        pass
    return None


def init_gemini_client() -> genai.Client:
    """Initialize Gemini client with support for multiple auth modes.

    Priority:
    1. GEMINI_API_KEY env var -> API key mode
    2. GCP_PROJECT env var -> Vertex AI mode
    3. project_id from credentials.json -> Vertex AI mode
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    gcp_project = os.environ.get("GCP_PROJECT", "").strip()
    gcp_location = os.environ.get("GCP_LOCATION", "us-central1").strip()

    if api_key:
        print("  Using Gemini API key mode")
        return genai.Client(api_key=api_key)

    if not gcp_project:
        gcp_project = _read_project_from_credentials() or ""

    if gcp_project:
        print(f"  Using Vertex AI mode (project: {gcp_project}, location: {gcp_location})")
        return genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location,
        )

    print("ERROR: No Gemini credentials configured.")
    print("Set one of the following in your .env file:")
    print("  GEMINI_API_KEY=your-api-key        (for API key mode)")
    print("  GCP_PROJECT=your-gcp-project-id    (for Vertex AI mode)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Gmail authentication (multi-account)
# ---------------------------------------------------------------------------

def _get_saved_accounts() -> dict[str, Path]:
    TOKENS_DIR.mkdir(exist_ok=True)
    accounts = {}
    for token_file in sorted(TOKENS_DIR.glob("token_*.json")):
        email = token_file.stem.removeprefix("token_")
        accounts[email] = token_file
    return accounts


def _get_email_from_service(service) -> str:
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]


def _load_or_refresh_creds(token_file: Path) -> Credentials | None:
    if not token_file.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open(token_file, "w") as f:
                f.write(creds.to_json())
            return creds
        except Exception:
            print(f"  Token refresh failed for {token_file.stem}. Will re-authenticate.")
            token_file.unlink(missing_ok=True)
    return None


def _run_oauth_flow() -> Credentials:
    if not CREDENTIALS_FILE.exists():
        print(f"ERROR: {CREDENTIALS_FILE} not found.")
        print("Please download your OAuth credentials from Google Cloud Console.")
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    try:
        return flow.run_local_server(port=0)
    except Exception as e:
        print(f"\nOAuth error: {e}")
        print("\nTroubleshooting:")
        print("  1. Wait 5-10 min after enabling the Gmail API")
        print("  2. Go to OAuth consent screen > Edit App > click through ALL 4 pages")
        print("  3. Confirm the Gmail account is listed under Test Users")
        print("  4. Try pasting the auth URL in an incognito browser window")
        sys.exit(1)


def authenticate_gmail():
    TOKENS_DIR.mkdir(exist_ok=True)
    accounts = _get_saved_accounts()

    if accounts:
        print("\nSaved Gmail accounts:")
        account_list = list(accounts.items())
        for i, (email, _) in enumerate(account_list, 1):
            print(f"  {i}. {email}")
        print(f"  {len(account_list) + 1}. Sign in to a new account")

        choice = input(f"\nSelect account (1-{len(account_list) + 1}): ").strip()

        try:
            idx = int(choice) - 1
        except ValueError:
            idx = -1

        if 0 <= idx < len(account_list):
            email, token_file = account_list[idx]
            print(f"\nLoading credentials for {email}...")
            creds = _load_or_refresh_creds(token_file)
            if creds:
                return build("gmail", "v1", credentials=creds)
            print("  Saved token expired. Signing in again...")

    print("\nOpening browser for Google sign-in...")
    print("(Choose whichever Gmail account you want to clean up)\n")
    creds = _run_oauth_flow()

    service = build("gmail", "v1", credentials=creds)
    email = _get_email_from_service(service)

    token_file = TOKENS_DIR / f"token_{email}.json"
    with open(token_file, "w") as f:
        f.write(creds.to_json())
    print(f"Saved credentials for {email}")

    return service

# ---------------------------------------------------------------------------
# Per-account data I/O
# ---------------------------------------------------------------------------

def _account_data_dir(email: str) -> Path:
    d = DATA_DIR / email
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_user_preferences(email: str) -> UserPreferences:
    path = _account_data_dir(email) / "user_preferences.json"
    if path.exists():
        with open(path, "r") as f:
            return UserPreferences.model_validate_json(f.read())
    return UserPreferences(account=email)


def save_user_preferences(email: str, prefs: UserPreferences) -> None:
    path = _account_data_dir(email) / "user_preferences.json"
    with open(path, "w") as f:
        f.write(prefs.model_dump_json(indent=2))


def load_feedback_log(email: str) -> list[FeedbackEntry]:
    path = _account_data_dir(email) / "feedback_log.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        return [FeedbackEntry.model_validate(entry) for entry in data]
    return []


def save_feedback_log(email: str, log: list[FeedbackEntry]) -> None:
    path = _account_data_dir(email) / "feedback_log.json"
    with open(path, "w") as f:
        json.dump([entry.model_dump() for entry in log], f, indent=2)


def load_sender_stats(email: str) -> dict[str, SenderStat]:
    path = _account_data_dir(email) / "sender_stats.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        return {k: SenderStat.model_validate(v) for k, v in data.items()}
    return {}


def save_sender_stats(email: str, stats: dict[str, SenderStat]) -> None:
    path = _account_data_dir(email) / "sender_stats.json"
    with open(path, "w") as f:
        json.dump({k: v.model_dump() for k, v in stats.items()}, f, indent=2)


def load_classification_cache(email: str) -> dict[str, dict]:
    path = _account_data_dir(email) / "cache_classifications.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_classification_cache(email: str, cache: dict[str, dict]) -> None:
    path = _account_data_dir(email) / "cache_classifications.json"
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def load_actions_log(email: str) -> list[ActionLogEntry]:
    path = _account_data_dir(email) / "actions_log.json"
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        return [ActionLogEntry.model_validate(entry) for entry in data]
    return []


def save_actions_log(email: str, log: list[ActionLogEntry]) -> None:
    path = _account_data_dir(email) / "actions_log.json"
    with open(path, "w") as f:
        json.dump([entry.model_dump() for entry in log], f, indent=2)

# ---------------------------------------------------------------------------
# Enhanced search (Stage 1)
# ---------------------------------------------------------------------------

def _parse_recent(recent_str: str) -> int | None:
    """Parse '30d' -> 30, '7d' -> 7, etc."""
    m = re.match(r"^(\d+)d$", recent_str.strip())
    return int(m.group(1)) if m else None


def build_search_query(
    keywords: list[str] | None = None,
    recent_days: int | None = None,
    label: str | None = None,
    unread_only: bool = False,
    from_domain: str | None = None,
) -> str:
    parts = []

    if keywords:
        kw_parts = []
        for kw in keywords:
            kw = kw.strip()
            if kw:
                kw_parts.append(f"from:({kw})")
                kw_parts.append(f"subject:({kw})")
        if kw_parts:
            parts.append("(" + " OR ".join(kw_parts) + ")")

    if recent_days:
        parts.append(f"newer_than:{recent_days}d")

    if label:
        parts.append(f"label:{label}")

    if unread_only:
        parts.append("is:unread")

    if from_domain:
        parts.append(f"from:@{from_domain}")

    return " ".join(parts)


def search_emails(service, query: str, max_results: int | None = None) -> list[dict]:
    print(f"\nSearching Gmail with query: {query}")

    all_message_ids = []
    page_token = None

    while True:
        try:
            results = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    maxResults=MAX_RESULTS_PER_PAGE,
                    pageToken=page_token,
                )
                .execute()
            )
        except HttpError as e:
            print(f"Error searching emails: {e}")
            break

        messages = results.get("messages", [])
        all_message_ids.extend(messages)

        page_token = results.get("nextPageToken")
        if not page_token:
            break

        print(f"  Found {len(all_message_ids)} emails so far, fetching more...")

    print(f"Found {len(all_message_ids)} matching emails total.")

    if not all_message_ids:
        return []

    # Apply max_results cap
    if max_results and len(all_message_ids) > max_results:
        print(f"  Capping to {max_results} most recent emails.")
        all_message_ids = all_message_ids[:max_results]
    elif len(all_message_ids) > 50 and max_results is None:
        cap = input(
            f"\n{len(all_message_ids)} emails matched. Process all, or enter a"
            " number to limit to the most recent N (default 50): "
        ).strip()
        if cap == "":
            all_message_ids = all_message_ids[:50]
        elif cap.lower() != "all":
            try:
                all_message_ids = all_message_ids[:int(cap)]
            except ValueError:
                all_message_ids = all_message_ids[:50]

    # Fetch metadata
    email_data = []
    for i, msg_stub in enumerate(all_message_ids):
        try:
            msg = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=msg_stub["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
        except HttpError as e:
            print(f"  Error fetching message {msg_stub['id']}: {e}")
            continue

        headers = {
            h["name"]: h["value"]
            for h in msg.get("payload", {}).get("headers", [])
        }

        email_data.append({
            "id": msg["id"],
            "sender": headers.get("From", "Unknown"),
            "subject": headers.get("Subject", "(no subject)"),
            "date": headers.get("Date", "Unknown"),
            "snippet": msg.get("snippet", ""),
        })

        if (i + 1) % 25 == 0:
            print(f"  Fetched metadata for {i + 1}/{len(all_message_ids)} emails...")

    print(f"Retrieved metadata for {len(email_data)} emails.")
    return email_data

# ---------------------------------------------------------------------------
# Rule-based pre-filtering (Stage 2)
# ---------------------------------------------------------------------------

def _extract_domain(sender: str) -> str:
    """Extract domain from 'Name <email@domain.com>' or 'email@domain.com'."""
    m = re.search(r"@([\w.-]+)", sender)
    return m.group(1).lower() if m else ""


def _extract_email_addr(sender: str) -> str:
    """Extract email address from 'Name <email@domain.com>'."""
    m = re.search(r"<([^>]+)>", sender)
    if m:
        return m.group(1).lower()
    if "@" in sender:
        return sender.strip().lower()
    return ""


def prefilter_emails(
    emails: list[dict],
    prefs: UserPreferences,
) -> tuple[list[dict], list[dict], list[dict]]:
    hard_keep = []
    hard_trash = []
    needs_gemini = []

    for email in emails:
        domain = _extract_domain(email["sender"])
        addr = _extract_email_addr(email["sender"])
        subj_lower = email["subject"].lower()
        snippet_lower = email.get("snippet", "").lower()

        # Hard keep: whitelisted
        if domain in prefs.whitelist_domains or addr in prefs.whitelist_senders:
            email["prefilter_decision"] = "keep"
            email["prefilter_reason"] = "Whitelisted sender/domain"
            hard_keep.append(email)
            continue

        # Hard keep: receipt keywords
        is_receipt = any(kw in subj_lower or kw in snippet_lower for kw in RECEIPT_KEYWORDS)
        if is_receipt and prefs.category_rules.get("receipt") == "keep":
            email["prefilter_decision"] = "keep"
            email["prefilter_reason"] = "Receipt/invoice detected"
            hard_keep.append(email)
            continue

        # Hard trash: blacklisted
        if domain in prefs.blacklist_domains:
            email["prefilter_decision"] = "trash"
            email["prefilter_reason"] = "Blacklisted domain"
            hard_trash.append(email)
            continue

        # Hard trash: always-trash patterns
        matched_pattern = False
        for pattern in prefs.always_trash_patterns:
            pat_lower = pattern.lower()
            if pat_lower in subj_lower or pat_lower in email["sender"].lower():
                email["prefilter_decision"] = "trash"
                email["prefilter_reason"] = f"Matches always-trash pattern: {pattern}"
                hard_trash.append(email)
                matched_pattern = True
                break
        if matched_pattern:
            continue

        # Needs Gemini
        email["prefilter_decision"] = None
        needs_gemini.append(email)

    return hard_keep, hard_trash, needs_gemini

# ---------------------------------------------------------------------------
# Dynamic prompt generation (Stage 3)
# ---------------------------------------------------------------------------

def build_dynamic_system_prompt(
    prefs: UserPreferences,
    sender_stats: dict[str, SenderStat],
    recent_feedback: list[FeedbackEntry],
    search_context: str,
    prompts_templates: dict,
) -> str:
    prompt = prompts_templates.get("system_prompt_base", "")

    # Preference context
    keep_cats = [c for c, r in prefs.category_rules.items() if r == "keep"]
    trash_cats = [c for c, r in prefs.category_rules.items() if r == "trash"]
    pref_template = prompts_templates.get("preference_context_template", "")
    if pref_template:
        prompt += pref_template.format(
            mode=prefs.mode,
            keep_categories=", ".join(keep_cats) if keep_cats else "none specified",
            trash_categories=", ".join(trash_cats) if trash_cats else "none specified",
            whitelist_domains=", ".join(prefs.whitelist_domains) if prefs.whitelist_domains else "none",
            blacklist_domains=", ".join(prefs.blacklist_domains) if prefs.blacklist_domains else "none",
        )

    # Sender stats context (top 20 by volume)
    if sender_stats:
        top_senders = sorted(sender_stats.values(),
                             key=lambda s: s.total_emails, reverse=True)[:20]
        stats_lines = []
        for s in top_senders:
            override_rate = (s.override_count / s.total_emails * 100) if s.total_emails > 0 else 0
            stats_lines.append(
                f"  - {s.sender_or_domain}: {s.total_emails} emails, "
                f"avg_importance={s.avg_importance:.2f}, avg_junk={s.avg_junk:.2f}, "
                f"override_rate={override_rate:.0f}%"
            )
        stats_template = prompts_templates.get("sender_stats_context_template", "")
        if stats_template and stats_lines:
            prompt += stats_template.format(
                sender_stats_block="\n".join(stats_lines)
            )

    # Feedback context (last 20)
    if recent_feedback:
        feedback_lines = []
        for fb in recent_feedback[-20:]:
            feedback_lines.append(
                f"  - Sender: {fb.sender[:40]}, Subject: {fb.subject[:40]}, "
                f"AI said: {fb.original_decision}, User corrected to: {fb.corrected_decision}"
                + (f" (reason: {fb.reason})" if fb.reason else "")
            )
        fb_template = prompts_templates.get("feedback_context_template", "")
        if fb_template and feedback_lines:
            prompt += fb_template.format(
                feedback_block="\n".join(feedback_lines)
            )

    # Search context
    search_template = prompts_templates.get("search_context_template", "")
    if search_template:
        prompt += search_template.format(search_description=search_context)

    return prompt

# ---------------------------------------------------------------------------
# Gemini classification (Stage 3)
# ---------------------------------------------------------------------------

def _build_emails_block(batch: list[dict], start_index: int) -> str:
    lines = []
    for i, email in enumerate(batch):
        idx = start_index + i
        lines.append(
            f"EMAIL {idx}:\n"
            f"  From: {email['sender']}\n"
            f"  Subject: {email['subject']}\n"
            f"  Date: {email['date']}\n"
            f"  Snippet: {email['snippet']}\n"
        )
    return "\n".join(lines)


def classify_batch(
    gemini_client: genai.Client,
    batch: list[dict],
    start_index: int,
    system_prompt: str,
    prompts_templates: dict,
) -> BatchClassification:
    emails_block = _build_emails_block(batch, start_index)
    user_prompt = prompts_templates.get("classification_prompt_v2", "").format(
        emails_block=emails_block,
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_json_schema": BatchClassification.model_json_schema(),
            "temperature": 0.2,
        },
    )

    return BatchClassification.model_validate_json(response.text)


def classify_all_emails(
    gemini_client: genai.Client,
    emails: list[dict],
    system_prompt: str,
    prompts_templates: dict,
    cache: dict[str, dict],
) -> list[dict]:
    # Check cache first
    uncached = []
    for email in emails:
        msg_id = email["id"]
        if msg_id in cache:
            cached = cache[msg_id]
            email["classification"] = EnhancedEmailClassification.model_validate(
                cached["classification"]
            )
            email["from_cache"] = True
        else:
            uncached.append(email)

    cached_count = len(emails) - len(uncached)
    if cached_count > 0:
        print(f"  {cached_count} emails loaded from cache, {len(uncached)} need classification")

    if not uncached:
        return emails

    # Classify uncached emails in batches
    total = len(uncached)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(num_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = uncached[start:end]

        print(f"\n  Batch {batch_num + 1}/{num_batches} "
              f"(emails {start + 1}-{end} of {total})...")

        max_retries = 5
        batch_result = None
        for attempt in range(max_retries):
            try:
                batch_result = classify_batch(
                    gemini_client, batch, start, system_prompt, prompts_templates
                )
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    match = re.search(r'retryDelay.*?(\d+)', str(e))
                    wait = int(match.group(1)) + 5 if match else 30
                    print(f"    Rate limited. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"    Error classifying batch: {e}")
                    break

        if batch_result:
            result_map = {c.email_index: c for c in batch_result.classifications}
            for i, email in enumerate(batch):
                idx = start + i
                if idx in result_map:
                    email["classification"] = result_map[idx]
                else:
                    email["classification"] = EnhancedEmailClassification(
                        email_index=idx,
                        category="unknown",
                        importance_score=0.5,
                        junk_score=0.0,
                        risk_of_wrong_deletion=1.0,
                        confidence=0.0,
                        reasoning="Not returned by model. Defaulting to keep.",
                    )
                # Write to cache
                cache[email["id"]] = {
                    "classification": email["classification"].model_dump(),
                    "timestamp": datetime.now().isoformat(),
                }
        else:
            for i, email in enumerate(batch):
                email["classification"] = EnhancedEmailClassification(
                    email_index=start + i,
                    category="unknown",
                    importance_score=0.5,
                    junk_score=0.0,
                    risk_of_wrong_deletion=1.0,
                    confidence=0.0,
                    reasoning="Batch classification failed. Defaulting to keep.",
                )

        if batch_num < num_batches - 1:
            time.sleep(RATE_LIMIT_DELAY)

    return emails

# ---------------------------------------------------------------------------
# Decision policy (Stage 4)
# ---------------------------------------------------------------------------

def apply_decision_policy(
    emails: list[dict],
    prefs: UserPreferences,
) -> list[dict]:
    keep_thresh = prefs.thresholds.get("keep", 0.75)
    trash_thresh = prefs.thresholds.get("trash", 0.85)
    risk_max = prefs.thresholds.get("risk_max", 0.2)

    # Mode adjustments
    if prefs.mode == "aggressive":
        trash_thresh = min(trash_thresh, 0.65)
        risk_max = max(risk_max, 0.35)
    elif prefs.mode == "balanced":
        trash_thresh = min(trash_thresh, 0.80)
        risk_max = max(risk_max, 0.25)

    for email in emails:
        c = email.get("classification")
        if not c:
            email["final_decision"] = "review"
            continue

        # Check category rules first
        cat_rule = prefs.category_rules.get(c.category)
        if cat_rule == "keep":
            email["final_decision"] = "keep"
            continue
        if cat_rule == "trash":
            email["final_decision"] = "trash"
            continue

        # Threshold-based decision
        if c.importance_score >= keep_thresh:
            email["final_decision"] = "keep"
        elif c.junk_score >= trash_thresh and c.risk_of_wrong_deletion <= risk_max:
            email["final_decision"] = "trash"
        else:
            email["final_decision"] = "review"

    return emails

# ---------------------------------------------------------------------------
# Active learning follow-up (Stage 5)
# ---------------------------------------------------------------------------

def select_followup_candidates(
    emails: list[dict],
    prefs: UserPreferences,
    sender_stats: dict[str, SenderStat],
    max_candidates: int = 8,
) -> list[dict]:
    candidates = []

    for email in emails:
        c = email.get("classification")
        if not c:
            continue

        domain = _extract_domain(email["sender"])
        score = 0.0

        # Uncertain (low confidence)
        if c.confidence < 0.6:
            score += 2.0

        # Near thresholds
        keep_thresh = prefs.thresholds.get("keep", 0.75)
        trash_thresh = prefs.thresholds.get("trash", 0.85)
        if abs(c.importance_score - keep_thresh) < 0.15:
            score += 1.0
        if abs(c.junk_score - trash_thresh) < 0.15:
            score += 1.0

        # New sender
        if domain and domain not in sender_stats:
            score += 1.5

        # Frequently overridden sender
        if domain in sender_stats:
            stat = sender_stats[domain]
            if stat.total_emails > 0 and stat.override_count / stat.total_emails > 0.3:
                score += 1.5

        if score >= 1.5:
            email["_followup_score"] = score
            candidates.append(email)

    candidates.sort(key=lambda e: e.get("_followup_score", 0), reverse=True)
    return candidates[:max_candidates]


def run_followup_questions(
    candidates: list[dict],
    feedback_log: list[FeedbackEntry],
    sender_stats: dict[str, SenderStat],
) -> tuple[list[dict], list[FeedbackEntry], dict[str, SenderStat]]:
    if not candidates:
        return candidates, feedback_log, sender_stats

    print(f"\n{'=' * 60}")
    print("ACTIVE LEARNING - Follow-Up Questions")
    print(f"{'=' * 60}")
    print(f"We'd like your feedback on {len(candidates)} uncertain email(s).\n")

    for email in candidates:
        c = email["classification"]
        decision = email.get("final_decision", "review")

        print(f"  Subject: {email['subject'][:60]}")
        print(f"  From:    {email['sender'][:50]}")
        print(f"  Category: {c.category} | Importance: {c.importance_score:.2f} | "
              f"Junk: {c.junk_score:.2f} | Risk: {c.risk_of_wrong_deletion:.2f}")
        print(f"  AI Decision: {decision.upper()}")
        print(f"  Reasoning: {c.reasoning[:80]}")

        answer = input("  Correct? (Y/n/skip): ").strip().lower()

        if answer == "skip" or answer == "s":
            print()
            continue

        if answer in ("n", "no"):
            correction = input("  Should this be (K)eep / (T)rash / (R)eview? ").strip().lower()
            if correction in ("k", "keep"):
                new_decision = "keep"
            elif correction in ("t", "trash"):
                new_decision = "trash"
            elif correction in ("r", "review"):
                new_decision = "review"
            else:
                print("  Invalid. Skipping.")
                print()
                continue

            reason = input("  Brief reason (enter to skip): ").strip()

            # Update the email's decision
            email["final_decision"] = new_decision

            # Log feedback
            feedback_log.append(FeedbackEntry(
                timestamp=datetime.now().isoformat(),
                message_id=email["id"],
                sender=email["sender"],
                subject=email["subject"],
                original_decision=decision,
                corrected_decision=new_decision,
                category=c.category,
                reason=reason,
            ))

            # Update sender stats
            domain = _extract_domain(email["sender"])
            if domain:
                if domain not in sender_stats:
                    sender_stats[domain] = SenderStat(sender_or_domain=domain)
                sender_stats[domain].override_count += 1

            print(f"  Updated to: {new_decision.upper()}")
        else:
            print("  Confirmed.")

        print()

    return candidates, feedback_log, sender_stats


def update_sender_stats_from_run(
    emails: list[dict],
    sender_stats: dict[str, SenderStat],
) -> dict[str, SenderStat]:
    for email in emails:
        domain = _extract_domain(email["sender"])
        if not domain:
            continue

        if domain not in sender_stats:
            sender_stats[domain] = SenderStat(sender_or_domain=domain)

        stat = sender_stats[domain]
        stat.total_emails += 1

        decision = email.get("final_decision", "review")
        if decision == "trash":
            stat.times_trashed += 1
        elif decision == "keep":
            stat.times_kept += 1
        else:
            stat.times_reviewed += 1

        c = email.get("classification")
        if c and hasattr(c, "importance_score"):
            n = stat.total_emails
            stat.avg_importance = ((stat.avg_importance * (n - 1)) + c.importance_score) / n
            stat.avg_junk = ((stat.avg_junk * (n - 1)) + c.junk_score) / n

    return sender_stats

# ---------------------------------------------------------------------------
# Action execution + safety
# ---------------------------------------------------------------------------

def ensure_label_exists(service, label_name: str = "GmailJanitor/Review") -> str:
    """Create the Gmail label if it doesn't exist. Returns the label ID."""
    try:
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])
        for lbl in labels:
            if lbl["name"] == label_name:
                return lbl["id"]

        # Create the label
        label_body = {
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
        created = service.users().labels().create(userId="me", body=label_body).execute()
        print(f"  Created Gmail label: {label_name}")
        return created["id"]
    except HttpError as e:
        print(f"  Warning: Could not create label '{label_name}': {e}")
        return ""


def execute_actions(
    service,
    emails: list[dict],
    review_label_id: str,
    no_trash: bool = False,
    dry_run: bool = False,
) -> dict:
    counts = {"kept": 0, "reviewed": 0, "trashed": 0, "errors": 0,
              "succeeded_ids": [], "failed_ids": []}

    for email in emails:
        decision = email.get("final_decision", "review")

        if decision == "keep":
            counts["kept"] += 1
            continue

        if decision == "review" or (decision == "trash" and no_trash):
            if dry_run:
                counts["reviewed"] += 1
                continue
            if review_label_id:
                try:
                    service.users().messages().modify(
                        userId="me",
                        id=email["id"],
                        body={"addLabelIds": [review_label_id]},
                    ).execute()
                    counts["reviewed"] += 1
                    counts["succeeded_ids"].append(email["id"])
                    print(f"  Labeled for review: {email['subject'][:50]}")
                except HttpError as e:
                    print(f"  Failed to label '{email['subject'][:40]}': {e}")
                    counts["errors"] += 1
                    counts["failed_ids"].append(email["id"])
            else:
                counts["reviewed"] += 1

        elif decision == "trash":
            if dry_run:
                counts["trashed"] += 1
                continue
            try:
                service.users().messages().trash(userId="me", id=email["id"]).execute()
                counts["trashed"] += 1
                counts["succeeded_ids"].append(email["id"])
                print(f"  Trashed: {email['subject'][:50]}")
            except HttpError as e:
                print(f"  Failed to trash '{email['subject'][:40]}': {e}")
                counts["errors"] += 1
                counts["failed_ids"].append(email["id"])

    return counts


def undo_last_action(service, email: str) -> None:
    actions_log = load_actions_log(email)
    if not actions_log:
        print("No actions to undo.")
        return

    last = actions_log[-1]
    if last.action == "undo":
        print("Last action was already an undo.")
        return

    print(f"\nUndoing action from {last.timestamp}:")
    print(f"  Query: {last.query_used}")
    print(f"  Counts: {last.counts}")

    # Untrash messages
    untrashed = 0
    for msg_id in last.message_ids:
        try:
            service.users().messages().untrash(userId="me", id=msg_id).execute()
            untrashed += 1
        except HttpError as e:
            print(f"  Failed to untrash {msg_id}: {e}")

    # Remove labels
    unlabeled = 0
    for msg_id in last.label_ids:
        # We don't track which label was applied, so we can't remove it generically
        # But we know it's the review label
        try:
            labels_result = service.users().labels().list(userId="me").execute()
            review_label_id = None
            for lbl in labels_result.get("labels", []):
                if "janitor" in lbl["name"].lower():
                    review_label_id = lbl["id"]
                    break
            if review_label_id:
                service.users().messages().modify(
                    userId="me",
                    id=msg_id,
                    body={"removeLabelIds": [review_label_id]},
                ).execute()
                unlabeled += 1
        except HttpError as e:
            print(f"  Failed to unlabel {msg_id}: {e}")

    print(f"\nUndo complete: {untrashed} untrashed, {unlabeled} unlabeled.")

    # Log the undo
    actions_log.append(ActionLogEntry(
        timestamp=datetime.now().isoformat(),
        action="undo",
        message_ids=last.message_ids,
        label_ids=last.label_ids,
        query_used=f"undo of {last.timestamp}",
    ))
    save_actions_log(email, actions_log)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_results(
    classified_emails: list[dict],
    prefilter_keep: list[dict],
    prefilter_trash: list[dict],
) -> None:
    all_decided = classified_emails + prefilter_keep + prefilter_trash

    trash_emails = [e for e in all_decided if e.get("final_decision") == "trash"]
    review_emails = [e for e in all_decided if e.get("final_decision") == "review"]
    keep_emails = [e for e in all_decided if e.get("final_decision") == "keep"]

    print(f"\n{'=' * 80}")
    print("CLASSIFICATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Total emails analyzed:  {len(all_decided)}")
    print(f"  Pre-filtered (keep):  {len(prefilter_keep)}")
    print(f"  Pre-filtered (trash): {len(prefilter_trash)}")
    print(f"  Gemini classified:    {len(classified_emails)}")
    print(f"{'=' * 80}")
    print(f"  TRASH:  {len(trash_emails)}")
    print(f"  REVIEW: {len(review_emails)}")
    print(f"  KEEP:   {len(keep_emails)}")
    print(f"{'=' * 80}\n")

    if trash_emails:
        print("--- EMAILS TO TRASH ---\n")
        table_data = []
        for i, email in enumerate(trash_emails, 1):
            c = email.get("classification")
            if c:
                table_data.append([
                    i,
                    email["subject"][:45],
                    email["sender"][:30],
                    c.category,
                    f"{c.junk_score:.2f}",
                    f"{c.risk_of_wrong_deletion:.2f}",
                    f"{c.confidence:.0%}",
                    c.reasoning[:50],
                ])
            else:
                reason = email.get("prefilter_reason", "Pre-filtered")
                table_data.append([
                    i, email["subject"][:45], email["sender"][:30],
                    "-", "-", "-", "-", reason[:50],
                ])
        print(tabulate(
            table_data,
            headers=["#", "Subject", "From", "Cat.", "Junk", "Risk", "Conf.", "Reasoning"],
            tablefmt="grid",
        ))

    if review_emails:
        print("\n--- EMAILS FOR REVIEW ---\n")
        table_data = []
        for i, email in enumerate(review_emails, 1):
            c = email.get("classification")
            if c:
                table_data.append([
                    i,
                    email["subject"][:45],
                    email["sender"][:30],
                    c.category,
                    f"{c.importance_score:.2f}",
                    f"{c.junk_score:.2f}",
                    f"{c.confidence:.0%}",
                ])
            else:
                table_data.append([
                    i, email["subject"][:45], email["sender"][:30],
                    "-", "-", "-", "-",
                ])
        print(tabulate(
            table_data,
            headers=["#", "Subject", "From", "Cat.", "Imp.", "Junk", "Conf."],
            tablefmt="grid",
        ))

    if keep_emails:
        print("\n--- EMAILS TO KEEP ---\n")
        table_data = []
        for email in keep_emails:
            c = email.get("classification")
            if c:
                table_data.append([
                    email["subject"][:50],
                    email["sender"][:35],
                    c.category,
                    f"{c.importance_score:.2f}",
                ])
            else:
                reason = email.get("prefilter_reason", "Pre-filtered")
                table_data.append([
                    email["subject"][:50], email["sender"][:35],
                    "-", reason[:20],
                ])
        print(tabulate(
            table_data,
            headers=["Subject", "From", "Cat.", "Importance"],
            tablefmt="simple",
        ))

# ---------------------------------------------------------------------------
# First-run preference wizard
# ---------------------------------------------------------------------------

def run_preference_wizard(prefs: UserPreferences) -> UserPreferences:
    print(f"\n{'=' * 60}")
    print("FIRST-RUN SETUP - Tell us about your email preferences")
    print(f"{'=' * 60}\n")

    # Mode
    print("How aggressive should cleanup be?")
    print("  1. Conservative (default: review uncertain emails, rarely trash)")
    print("  2. Balanced (trash high-confidence junk)")
    print("  3. Aggressive (trash medium-confidence junk)")
    mode_choice = input("Choose (1-3, default 1): ").strip()
    if mode_choice == "2":
        prefs.mode = "balanced"
    elif mode_choice == "3":
        prefs.mode = "aggressive"
    else:
        prefs.mode = "conservative"

    # Category preferences
    print("\nAre job alerts important to you?")
    job_answer = input("  (Y/n): ").strip().lower()
    if job_answer in ("n", "no"):
        prefs.category_rules["job_alert"] = "review"

    print("\nDo you want receipts always kept?")
    receipt_answer = input("  (Y/n, default Y): ").strip().lower()
    if receipt_answer in ("n", "no"):
        prefs.category_rules["receipt"] = "review"

    print("\nHow should newsletters be handled?")
    print("  1. Review (default)")
    print("  2. Trash")
    print("  3. Keep")
    nl_choice = input("Choose (1-3, default 1): ").strip()
    if nl_choice == "2":
        prefs.category_rules["marketing"] = "trash"
    elif nl_choice == "3":
        prefs.category_rules["marketing"] = "keep"

    print("\nPreferences saved!\n")
    return prefs

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  GMAIL JANITOR v2 - AI-Powered Email Cleanup")
    print("=" * 60)

    # Parse CLI args
    args = parse_cli_args()

    # Load environment
    load_dotenv(PROJECT_DIR / ".env")

    # Authenticate with Gmail
    print("\nAuthenticating with Gmail...")
    service = authenticate_gmail()
    email_address = _get_email_from_service(service)
    print(f"Authenticated as: {email_address}\n")

    # Handle --undo-last
    if args.undo_last:
        undo_last_action(service, email_address)
        return

    # Load prompts
    if not PROMPTS_FILE.exists():
        print(f"ERROR: {PROMPTS_FILE} not found.")
        sys.exit(1)
    prompts_templates = load_prompts()

    # Initialize Gemini client (portable)
    print("Initializing Gemini client...")
    gemini_client = init_gemini_client()

    # ── STAGE 0: Load user context ──
    prefs = load_user_preferences(email_address)
    feedback_log = load_feedback_log(email_address)
    sender_stats = load_sender_stats(email_address)
    cache = load_classification_cache(email_address)

    # First-run wizard
    prefs_path = _account_data_dir(email_address) / "user_preferences.json"
    if not prefs_path.exists() and not args.auto:
        prefs = run_preference_wizard(prefs)
        save_user_preferences(email_address, prefs)

    # Apply CLI overrides
    if args.mode:
        prefs.mode = args.mode
    if args.keep_threshold is not None:
        prefs.thresholds["keep"] = args.keep_threshold
    if args.trash_threshold is not None:
        prefs.thresholds["trash"] = args.trash_threshold
    if args.risk_max is not None:
        prefs.thresholds["risk_max"] = args.risk_max

    # ── STAGE 1: Build search query ──
    keywords = None
    if args.keywords:
        keywords = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]
    elif not args.auto and not args.recent and not args.label and not args.from_domain:
        print("\nSearch modes:")
        print("  1. Keywords (default)")
        print("  2. Recent emails")
        print("  3. Specific label")
        print("  4. From domain")
        search_mode = input("Choose (1-4, default 1): ").strip()

        if search_mode == "2":
            days = input("How many days back? (default 30): ").strip()
            args.recent = f"{days or '30'}d"
        elif search_mode == "3":
            args.label = input("Label name: ").strip()
        elif search_mode == "4":
            args.from_domain = input("Domain (e.g. linkedin.com): ").strip()
        else:
            keywords_input = input(
                "\nEnter keywords to search for (comma-separated).\n"
                "Examples: LinkedIn, Coursera, Best Buy, Uber Eats\n"
                "Keywords: "
            )
            if keywords_input.strip():
                keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]

    recent_days = _parse_recent(args.recent) if args.recent else None

    query = build_search_query(
        keywords=keywords,
        recent_days=recent_days,
        label=args.label,
        unread_only=args.unread_only,
        from_domain=args.from_domain,
    )

    if not query:
        print("No search criteria provided. Exiting.")
        sys.exit(0)

    emails = search_emails(service, query, max_results=args.max)
    if not emails:
        print("No matching emails found. Try different search criteria.")
        sys.exit(0)

    # ── STAGE 2: Rule-based pre-filtering ──
    hard_keep, hard_trash, needs_gemini = prefilter_emails(emails, prefs)
    print(f"\nPre-filter results:")
    print(f"  Auto-keep:  {len(hard_keep)}")
    print(f"  Auto-trash: {len(hard_trash)}")
    print(f"  Need AI:    {len(needs_gemini)}")

    # ── STAGE 3: Gemini classification ──
    classified = []
    if needs_gemini:
        print(f"\nClassifying {len(needs_gemini)} emails with Gemini 2.5 Flash...")
        system_prompt = build_dynamic_system_prompt(
            prefs, sender_stats, feedback_log[-20:], query, prompts_templates
        )
        classified = classify_all_emails(
            gemini_client, needs_gemini, system_prompt, prompts_templates, cache
        )

    # ── STAGE 4: Apply decision policy ──
    for e in hard_keep:
        e["final_decision"] = "keep"
    for e in hard_trash:
        e["final_decision"] = "trash"
    if classified:
        classified = apply_decision_policy(classified, prefs)

    all_emails = hard_keep + hard_trash + classified

    # ── Display results ──
    display_results(classified, hard_keep, hard_trash)

    # ── STAGE 5: Active learning follow-up ──
    if not args.auto and not args.dry_run and classified:
        candidates = select_followup_candidates(classified, prefs, sender_stats)
        if candidates:
            classified, feedback_log, sender_stats = run_followup_questions(
                candidates, feedback_log, sender_stats
            )

    # ── Update sender stats ──
    sender_stats = update_sender_stats_from_run(all_emails, sender_stats)

    # ── Execute actions ──
    review_label_id = ensure_label_exists(service, args.review_label)

    if args.dry_run:
        print(f"\n{'=' * 40}")
        print("[DRY RUN] No actions taken.")
        print(f"{'=' * 40}")
    else:
        # Show summary before confirmation
        trash_count = sum(1 for e in all_emails if e.get("final_decision") == "trash")
        review_count = sum(1 for e in all_emails if e.get("final_decision") == "review")
        keep_count = sum(1 for e in all_emails if e.get("final_decision") == "keep")

        print(f"\nReady to execute: {trash_count} trash, {review_count} label for review, "
              f"{keep_count} keep")

        if not args.auto:
            confirm = input("Proceed? (Y/n): ").strip().lower()
            if confirm and confirm not in ("y", "yes"):
                print("Cancelled. No actions taken.")
                # Still save data
                save_user_preferences(email_address, prefs)
                save_feedback_log(email_address, feedback_log)
                save_sender_stats(email_address, sender_stats)
                save_classification_cache(email_address, cache)
                return

        result_counts = execute_actions(
            service, all_emails, review_label_id,
            no_trash=args.no_trash, dry_run=False,
        )

        print(f"\nActions complete: {result_counts}")

        # Log the action
        trashed_ids = [e["id"] for e in all_emails if e.get("final_decision") == "trash"]
        labeled_ids = [e["id"] for e in all_emails if e.get("final_decision") == "review"]
        actions_log = load_actions_log(email_address)
        actions_log.append(ActionLogEntry(
            timestamp=datetime.now().isoformat(),
            action="cleanup",
            message_ids=trashed_ids,
            label_ids=labeled_ids,
            query_used=query,
            counts=result_counts,
        ))
        save_actions_log(email_address, actions_log)

    # ── Save all updated data ──
    save_user_preferences(email_address, prefs)
    save_feedback_log(email_address, feedback_log)
    save_sender_stats(email_address, sender_stats)
    save_classification_cache(email_address, cache)

    print("\nGmail Janitor v2 session complete.")


if __name__ == "__main__":
    main()

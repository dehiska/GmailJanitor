"""
Gmail Janitor v2.3 — Streamlit Front-End
Run with: .venv\\Scripts\\streamlit run app.py
"""

import ssl
import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path

# Import backend from main.py
from main import (
    PROJECT_DIR, TOKENS_DIR, GEMINI_MODEL,
    # Auth
    _get_saved_accounts, _load_or_refresh_creds, _run_oauth_flow,
    _get_email_from_service,
    # Data I/O
    load_user_preferences, save_user_preferences,
    load_feedback_log, save_feedback_log,
    load_sender_stats, save_sender_stats,
    load_classification_cache, save_classification_cache,
    load_actions_log, save_actions_log,
    _account_data_dir, update_sender_stats_from_run,
    # Search
    build_search_query, search_emails, _parse_recent,
    # Pre-filter
    prefilter_emails, _extract_domain,
    # Classification
    build_dynamic_system_prompt, classify_all_emails,
    # Decision
    apply_decision_policy,
    # Actions
    ensure_label_exists, execute_actions, undo_last_action,
    # Models
    UserPreferences, EnhancedEmailClassification, FeedbackEntry,
    SenderStat, ActionLogEntry, BatchClassification,
    # Gemini
    init_gemini_client, load_prompts,
)
from planner_service import parse_command, ActionPlan

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv(PROJECT_DIR / ".env")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gmail Janitor",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar CSS padding
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 20px;
        padding-right: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "authenticated": False,
    "service": None,
    "email_address": "",
    "gemini_client": None,
    "prefs": None,
    "prompts": None,
    "emails": [],
    "classified": [],
    "hard_keep": [],
    "hard_trash": [],
    "plan_result": None,
    "selected_ids": set(),
    "quarantine_emails": [],
    "labels_map": {},
    "review_label": "GmailJanitor/Review",
    "command_scope": "Entire mailbox",
    "total_matching": 0,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# SSL-safe Gmail API wrapper
# ---------------------------------------------------------------------------
def safe_gmail_call(fn, *args, **kwargs):
    """Wrap a Gmail API call with SSL error recovery."""
    try:
        return fn(*args, **kwargs)
    except (ssl.SSLError, OSError, ConnectionError, Exception) as e:
        err_str = str(e).lower()
        if any(kw in err_str for kw in ["ssl", "wrong_version", "connection", "eof", "reset"]):
            st.error("Gmail connection failed (SSL/network issue).")
            col1, col2, col3 = st.columns(3)
            if col1.button("🔄 Retry", key="ssl_retry"):
                st.rerun()
            if col2.button("🔑 Re-authenticate", key="ssl_reauth"):
                st.session_state.service = None
                st.session_state.authenticated = False
                st.session_state.labels_map = {}
                st.rerun()
            if col3.button("🚪 Sign out", key="ssl_signout"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
            st.stop()
        raise


# ---------------------------------------------------------------------------
# Helper: Gmail auth for Streamlit
# ---------------------------------------------------------------------------
def streamlit_authenticate(account_email: str | None = None):
    TOKENS_DIR.mkdir(exist_ok=True)
    accounts = _get_saved_accounts()

    if account_email and account_email in accounts:
        token_file = accounts[account_email]
        creds = _load_or_refresh_creds(token_file)
        if creds:
            service = build("gmail", "v1", credentials=creds)
            st.session_state.service = service
            st.session_state.email_address = account_email
            st.session_state.authenticated = True
            return True
        else:
            st.warning(f"Token expired for {account_email}. Please sign in again.")

    try:
        creds = _run_oauth_flow()
        service = build("gmail", "v1", credentials=creds)
        email = _get_email_from_service(service)
        token_file = TOKENS_DIR / f"token_{email}.json"
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        st.session_state.service = service
        st.session_state.email_address = email
        st.session_state.authenticated = True
        return True
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False


def init_gemini():
    if st.session_state.gemini_client is None:
        try:
            st.session_state.gemini_client = init_gemini_client()
        except SystemExit:
            st.error("No Gemini credentials configured. Set GEMINI_API_KEY or GCP_PROJECT in .env")
            return False
    if st.session_state.prompts is None:
        st.session_state.prompts = load_prompts()
    return True


# ---------------------------------------------------------------------------
# Readable timestamps
# ---------------------------------------------------------------------------
def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        now = datetime.now()
        delta = now - dt
        if delta.total_seconds() < 60:
            relative = "just now"
        elif delta.total_seconds() < 3600:
            mins = int(delta.total_seconds() / 60)
            relative = f"{mins} min ago"
        elif delta.total_seconds() < 86400:
            hrs = int(delta.total_seconds() / 3600)
            relative = f"{hrs} hr ago"
        else:
            days = delta.days
            relative = f"{days} day{'s' if days != 1 else ''} ago"
        formatted = dt.strftime("%b %d, %Y • %I:%M %p")
        return f"{formatted} ({relative})"
    except Exception:
        return ts


# ---------------------------------------------------------------------------
# Helper: Email card rendering
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "job_alert": "🟢", "marketing": "🟠", "receipt": "🔵", "financial": "🟣",
    "social": "🟡", "personal": "🟢", "system": "⚪", "unknown": "⚫",
}

# Issue #6: Renamed decision badges to show these are RECOMMENDATIONS, not current state
DECISION_LABELS = {
    "keep": ("green", "→ KEEP"),
    "review": ("orange", "→ REVIEW"),
    "trash": ("red", "→ TRASH"),
}


def render_email_card(email: dict, idx: int, show_actions: bool = True,
                      key_prefix: str = "", on_override=None):
    """Render an email card. on_override(email, new_decision) is called when user clicks an action button."""
    c = email.get("classification")
    decision = email.get("final_decision", email.get("prefilter_decision", "review"))
    prefilter_reason = email.get("prefilter_reason", "")

    with st.container(border=True):
        col1, col2, col3 = st.columns([5, 2, 1])
        with col1:
            st.markdown(f"**{email['subject'][:80]}**")
            st.caption(f"From: {email['sender'][:60]}  ·  {email.get('date', '')[:25]}")
        with col2:
            if c:
                cat_icon = CATEGORY_COLORS.get(c.category, "⚫")
                st.markdown(f"{cat_icon} `{c.category}`")
            elif prefilter_reason:
                st.caption(f"Pre-filtered: {prefilter_reason}")
        with col3:
            color, label_text = DECISION_LABELS.get(decision, ("gray", decision.upper()))
            st.markdown(
                f"<span style='background-color:{color};color:white;padding:2px 8px;"
                f"border-radius:4px;font-size:0.8em'>{label_text}</span>",
                unsafe_allow_html=True,
            )

        if c:
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Importance", f"{c.importance_score:.2f}")
            s2.metric("Junk", f"{c.junk_score:.2f}")
            s3.metric("Risk", f"{c.risk_of_wrong_deletion:.2f}")
            s4.metric("Confidence", f"{c.confidence:.0%}")

        if email.get("snippet"):
            st.code(email["snippet"], language=None)

        if c and c.reasoning:
            with st.expander("Reasoning"):
                st.write(c.reasoning)

        if show_actions:
            a1, a2, a3, a4 = st.columns(4)
            key = f"{key_prefix}_{idx}"
            if a1.button("✅ Keep", key=f"keep_{key}", use_container_width=True):
                old_decision = email.get("final_decision", "review")
                email["final_decision"] = "keep"
                if on_override and old_decision != "keep":
                    on_override(email, "keep", old_decision)
                st.rerun()
            if a2.button("🔍 Review", key=f"review_{key}", use_container_width=True):
                old_decision = email.get("final_decision", "review")
                email["final_decision"] = "review"
                if on_override and old_decision != "review":
                    on_override(email, "review", old_decision)
                st.rerun()
            if a3.button("🗑️ Trash", key=f"trash_{key}", use_container_width=True):
                old_decision = email.get("final_decision", "review")
                email["final_decision"] = "trash"
                if on_override and old_decision != "trash":
                    on_override(email, "trash", old_decision)
                st.rerun()
            msg_id = email.get("id", "")
            if msg_id:
                gmail_url = f"https://mail.google.com/mail/u/0/#all/{msg_id}"
                a4.link_button("📧 Open", gmail_url, use_container_width=True)


def render_email_list(emails: list[dict], show_actions: bool = True,
                      key_prefix: str = "", page_size: int = 25, on_override=None):
    if not emails:
        st.info("No emails to display.")
        return

    total = len(emails)
    page_key = f"page_{key_prefix}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    start = st.session_state[page_key] * page_size
    end = min(start + page_size, total)
    page_emails = emails[start:end]

    st.caption(f"Showing {start + 1}-{end} of {total} emails")

    for i, email in enumerate(page_emails):
        render_email_card(email, start + i, show_actions=show_actions,
                          key_prefix=key_prefix, on_override=on_override)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if start > 0:
            if st.button("← Previous", key=f"prev_{key_prefix}"):
                st.session_state[page_key] -= 1
                st.rerun()
    with col2:
        total_pages = (total + page_size - 1) // page_size
        st.caption(f"Page {st.session_state[page_key] + 1} of {total_pages}")
    with col3:
        if end < total:
            if st.button("Next →", key=f"next_{key_prefix}"):
                st.session_state[page_key] += 1
                st.rerun()


# ---------------------------------------------------------------------------
# Helper: Fetch emails from a label (with SSL guard)
# ---------------------------------------------------------------------------
def fetch_label_emails(service, label_id: str, max_results: int = 50) -> list[dict]:
    try:
        results = safe_gmail_call(
            service.users().messages().list(userId="me", labelIds=[label_id], maxResults=max_results).execute
        )
    except HttpError as e:
        st.error(f"Error fetching label emails: {e}")
        return []

    messages = results.get("messages", [])
    if not messages:
        return []

    email_data = []
    progress = st.progress(0, text="Fetching email metadata...")
    for i, msg_stub in enumerate(messages):
        try:
            msg = service.users().messages().get(
                userId="me", id=msg_stub["id"],
                format="metadata", metadataHeaders=["From", "Subject", "Date"],
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            email_data.append({
                "id": msg["id"],
                "sender": headers.get("From", "Unknown"),
                "subject": headers.get("Subject", "(no subject)"),
                "date": headers.get("Date", "Unknown"),
                "snippet": msg.get("snippet", ""),
                "label_ids": msg.get("labelIds", []),
            })
        except (HttpError, Exception):
            pass
        progress.progress((i + 1) / len(messages))
    progress.empty()
    return email_data


def get_gmail_labels(service) -> dict:
    try:
        results = safe_gmail_call(
            service.users().labels().list(userId="me").execute
        )
        return {lbl["name"]: lbl["id"] for lbl in results.get("labels", [])}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Log helper that tracks truth (succeeded vs attempted)
# ---------------------------------------------------------------------------
def log_action(email_addr: str, action: str, query: str, counts: dict, dry_run: bool):
    actions_log = load_actions_log(email_addr)
    succeeded = counts.get("succeeded_ids", [])
    failed = counts.get("failed_ids", [])
    attempted = succeeded + failed

    actions_log.append(ActionLogEntry(
        timestamp=datetime.now().isoformat(),
        action=action,
        message_ids=[mid for mid in succeeded if counts.get("trashed", 0) > 0],
        label_ids=[mid for mid in succeeded if counts.get("reviewed", 0) > 0],
        query_used=query,
        counts={"kept": counts.get("kept", 0), "reviewed": counts.get("reviewed", 0),
                "trashed": counts.get("trashed", 0), "errors": counts.get("errors", 0)},
        attempted_ids=attempted,
        succeeded_ids=succeeded,
        failed_ids=failed,
        dry_run=dry_run,
    ))
    save_actions_log(email_addr, actions_log)


# ---------------------------------------------------------------------------
# Issue #7: Feedback logging when user overrides AI decision
# ---------------------------------------------------------------------------
def log_feedback(email: dict, new_decision: str, old_decision: str):
    """Log feedback when user overrides an AI classification."""
    email_addr = st.session_state.get("email_address", "")
    if not email_addr:
        return
    c = email.get("classification")
    feedback_log = load_feedback_log(email_addr)
    feedback_log.append(FeedbackEntry(
        timestamp=datetime.now().isoformat(),
        message_id=email.get("id", ""),
        sender=email.get("sender", "Unknown"),
        subject=email.get("subject", "(no subject)"),
        original_decision=old_decision,
        corrected_decision=new_decision,
        category=c.category if c else "unknown",
        reason="User override via Streamlit UI",
    ))
    save_feedback_log(email_addr, feedback_log)

    # Also update sender stats override count
    domain = _extract_domain(email.get("sender", ""))
    if domain:
        sender_stats = load_sender_stats(email_addr)
        if domain not in sender_stats:
            sender_stats[domain] = SenderStat(sender_or_domain=domain)
        sender_stats[domain].override_count += 1
        save_sender_stats(email_addr, sender_stats)


# ---------------------------------------------------------------------------
# SIDEBAR (Issue #1: removed dry-run toggle)
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("🧹 Gmail Janitor")

        # --- Account selector ---
        st.subheader("Account")
        accounts = _get_saved_accounts()

        if st.session_state.authenticated:
            st.success(f"✅ {st.session_state.email_address}")
            if st.button("Sign out", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        else:
            if accounts:
                account_list = list(accounts.keys())
                selected = st.radio("Select account:", account_list, key="account_select")
                if st.button("Connect", use_container_width=True, type="primary"):
                    with st.spinner("Authenticating..."):
                        streamlit_authenticate(selected)
                    st.rerun()
            if st.button("Sign in to new account", use_container_width=True):
                with st.spinner("Opening browser for sign-in..."):
                    streamlit_authenticate(None)
                st.rerun()

        if not st.session_state.authenticated:
            return

        st.divider()

        # --- Load preferences ---
        email = st.session_state.email_address
        if st.session_state.prefs is None:
            st.session_state.prefs = load_user_preferences(email)
        prefs = st.session_state.prefs

        # --- Mode selector ---
        st.subheader("Settings")
        mode_options = ["conservative", "balanced", "aggressive"]
        mode_idx = mode_options.index(prefs.mode) if prefs.mode in mode_options else 0
        new_mode = st.selectbox("Mode", mode_options, index=mode_idx)
        if new_mode != prefs.mode:
            prefs.mode = new_mode

        with st.expander("Thresholds"):
            prefs.thresholds["keep"] = st.slider(
                "Keep threshold", 0.0, 1.0, prefs.thresholds.get("keep", 0.75), 0.05,
                help="Emails with importance >= this are kept"
            )
            prefs.thresholds["trash"] = st.slider(
                "Trash threshold", 0.0, 1.0, prefs.thresholds.get("trash", 0.85), 0.05,
                help="Emails with junk score >= this are trashed"
            )
            prefs.thresholds["risk_max"] = st.slider(
                "Max risk", 0.0, 1.0, prefs.thresholds.get("risk_max", 0.2), 0.05,
                help="Max risk_of_wrong_deletion to allow trashing"
            )

        # --- Quarantine label ---
        labels = get_gmail_labels(st.session_state.service) if st.session_state.service else {}
        label_names = sorted(labels.keys()) if labels else ["GmailJanitor/Review"]
        default_label = "GmailJanitor/Review"
        if default_label in label_names:
            default_idx = label_names.index(default_label)
        else:
            default_idx = 0
        st.session_state.review_label = st.selectbox(
            "Quarantine label", label_names, index=default_idx
        )
        st.session_state.labels_map = labels

        st.divider()

        # --- Command box + scope selector + examples ---
        st.subheader("🗣️ Command")

        examples = [
            "Trash all marketing emails from last 30 days",
            "Keep all job alerts from LinkedIn",
            "Undo my last cleanup",
        ]
        ex_cols = st.columns(len(examples))
        for i, ex in enumerate(examples):
            if ex_cols[i].button(ex[:18] + "…", key=f"ex_{i}", use_container_width=True):
                st.session_state.nl_command = ex

        command = st.text_area(
            "Tell Gmail Janitor what to do...",
            placeholder="e.g. Trash all Red Cross marketing emails older than 30 days",
            height=160,
            key="nl_command",
        )

        # Issue #2: Scope selector — "Entire mailbox" is now the default
        scope = st.selectbox(
            "Default scope (if command is broad)",
            ["Entire mailbox", "Recent 7d", "Recent 30d", "Recent 90d",
             f"Label: {st.session_state.review_label}"],
            index=0,
            key="command_scope_select",
        )
        st.session_state.command_scope = scope

        c1, c2 = st.columns(2)
        if c1.button("Preview Plan", use_container_width=True, disabled=not command):
            if init_gemini():
                with st.spinner("Parsing command..."):
                    try:
                        plan = parse_command(
                            st.session_state.gemini_client, command,
                            GEMINI_MODEL, prefs.model_dump(),
                        )
                        st.session_state.plan_result = plan
                    except Exception as e:
                        st.error(f"Failed to parse command: {e}")

        if c2.button("Execute", use_container_width=True, type="primary",
                      disabled=st.session_state.plan_result is None):
            execute_plan()

        if st.session_state.plan_result:
            plan = st.session_state.plan_result
            st.info(f"**Plan:** {plan.explanation}")
            if plan.needs_scope_confirmation:
                st.warning(f"Broad command — using scope: **{scope}**")
            if plan.actions:
                for action in plan.actions:
                    st.caption(
                        f"• {action.action_type}"
                        + (f" (category: {action.filter_category})" if action.filter_category else "")
                        + (f" (from: {action.filter_from_domain})" if action.filter_from_domain else "")
                    )
            if plan.preference_updates.category_rules:
                st.caption(f"Preference changes: {plan.preference_updates.category_rules}")

        st.divider()
        if st.button("💾 Save preferences", use_container_width=True):
            save_user_preferences(email, prefs)
            st.success("Preferences saved!")


# ---------------------------------------------------------------------------
# Execute NL command plan (Issue #2: respect "Entire mailbox" scope)
# ---------------------------------------------------------------------------
def execute_plan():
    plan = st.session_state.plan_result
    if not plan:
        return

    service = st.session_state.service
    email = st.session_state.email_address
    prefs = st.session_state.prefs

    if plan.is_undo:
        with st.spinner("Undoing last action..."):
            undo_last_action(service, email)
        st.success("Undo complete!")
        st.session_state.plan_result = None
        return

    # Apply preference updates
    if plan.preference_updates.category_rules:
        for cat, rule in plan.preference_updates.category_rules.items():
            prefs.category_rules[cat] = rule
    for d in plan.preference_updates.whitelist_domains:
        if d not in prefs.whitelist_domains:
            prefs.whitelist_domains.append(d)
    for d in plan.preference_updates.blacklist_domains:
        if d not in prefs.blacklist_domains:
            prefs.blacklist_domains.append(d)
    for p in plan.preference_updates.always_trash_patterns:
        if p not in prefs.always_trash_patterns:
            prefs.always_trash_patterns.append(p)
    save_user_preferences(email, prefs)

    # Issue #2: Apply scope override — ALWAYS respect the user's scope choice
    search = plan.search
    scope = st.session_state.get("command_scope", "Entire mailbox")
    if scope == "Entire mailbox":
        # User explicitly wants entire mailbox — remove any time restriction
        search.recent_days = 0
    elif scope.startswith("Recent"):
        days = int(scope.split()[1].rstrip("d"))
        search.recent_days = days
    elif scope.startswith("Label:"):
        search.label = scope.split(":", 1)[1].strip()
        search.recent_days = 0

    # Issue #3: Remove max_results cap for commands — fetch ALL matching emails
    search.max_results = 0  # 0 = no cap

    query = build_search_query(
        keywords=search.keywords if search.keywords else None,
        recent_days=search.recent_days if search.recent_days > 0 else None,
        label=search.label if search.label else None,
        unread_only=search.unread_only,
        from_domain=search.from_domain if search.from_domain else None,
    )
    if search.raw_query:
        query = search.raw_query + (" " + query if query else "")

    if not query:
        query = "newer_than:30d"

    with st.spinner("Searching emails..."):
        emails_found = search_emails(service, query, max_results=None)

    if not emails_found:
        st.info("No emails matched the search.")
        st.session_state.plan_result = None
        return

    st.success(f"Found **{len(emails_found)}** matching emails")

    # Apply action filters
    for email_item in emails_found:
        domain = _extract_domain(email_item["sender"])
        for action in plan.actions:
            match = True
            if action.filter_from_domain and action.filter_from_domain.lower() not in domain:
                match = False
            if action.filter_subject_contains and action.filter_subject_contains.lower() not in email_item["subject"].lower():
                match = False
            if match:
                email_item["final_decision"] = action.action_type
                break
        if "final_decision" not in email_item:
            email_item["final_decision"] = "review"

    trash_n = sum(1 for e in emails_found if e.get("final_decision") == "trash")
    review_n = sum(1 for e in emails_found if e.get("final_decision") == "review")
    keep_n = sum(1 for e in emails_found if e.get("final_decision") == "keep")

    st.info(f"Will trash **{trash_n}**, review **{review_n}**, keep **{keep_n}** of **{len(emails_found)}** emails")

    # Confirmation before executing
    confirm_key = f"confirm_execute_{len(emails_found)}"
    if trash_n > 0:
        st.warning(f"⚠️ This will trash **{trash_n}** emails. This is reversible (Gmail Trash).")

    review_label_id = ensure_label_exists(service, st.session_state.review_label)
    with st.spinner(f"Executing actions on {len(emails_found)} emails..."):
        counts = execute_actions(service, emails_found, review_label_id, dry_run=False)
    succ = len(counts.get("succeeded_ids", []))
    fail = len(counts.get("failed_ids", []))
    st.success(f"Done! {succ} succeeded, {fail} failed. "
               f"(Trashed: {counts['trashed']}, Reviewed: {counts['reviewed']})")
    log_action(email, "command", query, counts, dry_run=False)

    st.session_state.plan_result = None


# ---------------------------------------------------------------------------
# TAB 1: Run Cleanup (Issues #3, #4, #5: no cap, select all, remove after trash)
# ---------------------------------------------------------------------------
def tab_run_cleanup():
    st.header("Run Cleanup")

    if not st.session_state.authenticated:
        st.warning("Please sign in from the sidebar first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        search_mode = st.radio("Search mode", ["Keywords", "Recent", "Label", "From domain"],
                               horizontal=True)
    with col2:
        max_results = st.number_input("Max emails to process", min_value=1, max_value=5000,
                                      value=100,
                                      help="Set high to process all. Gmail API fetches in pages of 100.")

    keywords = None
    recent_days = None
    label = None
    from_domain = None
    unread_only = st.checkbox("Unread only")

    if search_mode == "Keywords":
        kw_input = st.text_input("Keywords (comma-separated)", placeholder="LinkedIn, Coursera, Best Buy")
        if kw_input:
            keywords = [kw.strip() for kw in kw_input.split(",") if kw.strip()]
    elif search_mode == "Recent":
        days = st.number_input("Days back", min_value=1, max_value=365, value=30)
        recent_days = days
    elif search_mode == "Label":
        labels = st.session_state.get("labels_map", {})
        if labels:
            label = st.selectbox("Select label", sorted(labels.keys()), key="cleanup_label")
        else:
            label = st.text_input("Label name")
    elif search_mode == "From domain":
        from_domain = st.text_input("Domain", placeholder="linkedin.com")

    col1, col2 = st.columns(2)
    run_preview = col1.button("🔍 Preview Classification", type="primary", use_container_width=True)
    run_execute = col2.button("🚀 Execute Actions", use_container_width=True,
                              disabled=not st.session_state.classified and not st.session_state.hard_keep and not st.session_state.hard_trash)

    if run_preview:
        if not init_gemini():
            return

        query = build_search_query(
            keywords=keywords, recent_days=recent_days, label=label,
            unread_only=unread_only, from_domain=from_domain,
        )
        if not query:
            st.warning("No search criteria. Enter keywords or select a search mode.")
            return

        service = st.session_state.service
        prefs = st.session_state.prefs
        email_addr = st.session_state.email_address

        with st.spinner("Searching Gmail..."):
            emails = search_emails(service, query, max_results=max_results)

        if not emails:
            st.info("No matching emails found.")
            return

        st.session_state.total_matching = len(emails)
        st.success(f"Found **{len(emails)}** emails matching your search")

        hard_keep, hard_trash, needs_gemini = prefilter_emails(emails, prefs)
        st.info(f"Pre-filter: {len(hard_keep)} auto-keep, {len(hard_trash)} auto-trash, "
                f"{len(needs_gemini)} need AI")

        classified = []
        if needs_gemini:
            feedback_log = load_feedback_log(email_addr)
            sender_stats = load_sender_stats(email_addr)
            cache = load_classification_cache(email_addr)
            system_prompt = build_dynamic_system_prompt(
                prefs, sender_stats, feedback_log[-20:], query, st.session_state.prompts,
            )
            with st.spinner(f"Classifying {len(needs_gemini)} emails with Gemini..."):
                classified = classify_all_emails(
                    st.session_state.gemini_client, needs_gemini,
                    system_prompt, st.session_state.prompts, cache,
                )
            save_classification_cache(email_addr, cache)

        for e in hard_keep:
            e["final_decision"] = "keep"
        for e in hard_trash:
            e["final_decision"] = "trash"
        if classified:
            classified = apply_decision_policy(classified, prefs)

        st.session_state.emails = emails
        st.session_state.classified = classified
        st.session_state.hard_keep = hard_keep
        st.session_state.hard_trash = hard_trash
        st.rerun()

    # Display results
    classified = st.session_state.classified
    hard_keep = st.session_state.hard_keep
    hard_trash = st.session_state.hard_trash

    if classified or hard_keep or hard_trash:
        all_emails = hard_keep + hard_trash + classified
        trash_list = [e for e in all_emails if e.get("final_decision") == "trash"]
        review_list = [e for e in all_emails if e.get("final_decision") == "review"]
        keep_list = [e for e in all_emails if e.get("final_decision") == "keep"]

        total_matching = st.session_state.get("total_matching", len(all_emails))

        # Issue #3: Show total matching count
        st.caption(f"Processed {len(all_emails)} of {total_matching} total matching emails")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(all_emails))
        m2.metric("→ Trash", len(trash_list))
        m3.metric("→ Review", len(review_list))
        m4.metric("→ Keep", len(keep_list))

        # Issue #4: Bulk actions — Select All + Trash All / Keep All
        st.divider()
        st.subheader("Bulk Actions")
        st.caption("These apply the selected action to ALL emails in the results (not just the current page).")

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        if bcol1.button(f"🗑️ Trash All ({len(all_emails)})", use_container_width=True,
                        type="primary"):
            for e in all_emails:
                e["final_decision"] = "trash"
            st.rerun()
        if bcol2.button(f"🗑️ Trash Recommended ({len(trash_list)})", use_container_width=True):
            pass  # Already marked as trash
        if bcol3.button(f"✅ Keep All ({len(all_emails)})", use_container_width=True):
            for e in all_emails:
                e["final_decision"] = "keep"
            st.rerun()
        if bcol4.button("🔄 Reset Decisions", use_container_width=True):
            # Re-apply the policy
            prefs = st.session_state.prefs
            for e in hard_keep:
                e["final_decision"] = "keep"
            for e in hard_trash:
                e["final_decision"] = "trash"
            if classified:
                apply_decision_policy(classified, prefs)
            st.rerun()

        st.divider()

        t_trash, t_review, t_keep = st.tabs(["🗑️ Trash", "🔍 Review", "✅ Keep"])
        with t_trash:
            render_email_list(trash_list, key_prefix="trash", on_override=log_feedback)
        with t_review:
            render_email_list(review_list, key_prefix="review", on_override=log_feedback)
        with t_keep:
            render_email_list(keep_list, show_actions=False, key_prefix="keep")

    # Issue #5: Execute actions and remove trashed from list
    if run_execute and (classified or hard_keep or hard_trash):
        all_emails = hard_keep + hard_trash + classified
        service = st.session_state.service
        email_addr = st.session_state.email_address

        trash_count = sum(1 for e in all_emails if e.get("final_decision") == "trash")
        review_count = sum(1 for e in all_emails if e.get("final_decision") == "review")

        if trash_count == 0 and review_count == 0:
            st.info("Nothing to execute — all emails are marked 'keep'.")
        else:
            review_label_id = ensure_label_exists(service, st.session_state.review_label)
            with st.spinner(f"Executing: trashing {trash_count}, labeling {review_count}..."):
                counts = execute_actions(service, all_emails, review_label_id, dry_run=False)
            succ = len(counts.get("succeeded_ids", []))
            fail = len(counts.get("failed_ids", []))
            st.success(f"Done! {succ} succeeded, {fail} failed. "
                       f"(Trashed: {counts['trashed']}, Reviewed: {counts['reviewed']})")

            log_action(email_addr, "cleanup", "streamlit_cleanup", counts, dry_run=False)

            sender_stats = load_sender_stats(email_addr)
            sender_stats = update_sender_stats_from_run(all_emails, sender_stats)
            save_sender_stats(email_addr, sender_stats)

            # Issue #5: Remove succeeded emails from the displayed lists
            succeeded_set = set(counts.get("succeeded_ids", []))
            if succeeded_set:
                st.session_state.classified = [
                    e for e in st.session_state.classified if e["id"] not in succeeded_set
                ]
                st.session_state.hard_trash = [
                    e for e in st.session_state.hard_trash if e["id"] not in succeeded_set
                ]
                st.session_state.hard_keep = [
                    e for e in st.session_state.hard_keep if e["id"] not in succeeded_set
                ]
                st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: Quarantine
# ---------------------------------------------------------------------------
def tab_quarantine():
    st.header("Quarantine / Label Browser")

    if not st.session_state.authenticated:
        st.warning("Please sign in from the sidebar first.")
        return

    service = st.session_state.service
    labels_map = st.session_state.get("labels_map", {})

    col1, col2 = st.columns([3, 1])
    with col1:
        review_label = st.session_state.review_label
        priority = ["INBOX", "STARRED", "SENT", "GmailJanitor/Review",
                     "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES"]
        sorted_labels = [l for l in priority if l in labels_map]
        sorted_labels += [l for l in sorted(labels_map.keys()) if l not in sorted_labels]

        selected_label = st.selectbox(
            "Browse label", sorted_labels or [review_label],
            index=sorted_labels.index(review_label) if review_label in sorted_labels else 0,
            key="quarantine_label_select",
        )
    with col2:
        max_fetch = st.number_input("Max emails", 10, 500, 50, key="quarantine_max")

    fcol1, fcol2, fcol3 = st.columns(3)
    filter_unread = fcol1.checkbox("Unread only", key="q_unread")
    filter_category = fcol2.selectbox(
        "Category filter",
        ["All", "marketing", "social", "job_alert", "receipt", "financial", "personal", "system", "unknown"],
        key="q_cat_filter",
    )
    filter_search = fcol3.text_input("Search subject/from", key="q_search")

    if st.button("📥 Load Emails", type="primary", use_container_width=True):
        label_id = labels_map.get(selected_label, selected_label)
        with st.spinner(f"Fetching emails from '{selected_label}'..."):
            q_emails = fetch_label_emails(service, label_id, max_results=max_fetch)
        st.session_state.quarantine_emails = q_emails
        st.session_state.selected_ids = set()

    q_emails = st.session_state.get("quarantine_emails", [])
    if not q_emails:
        st.info("Click 'Load Emails' to fetch emails from the selected label.")
        return

    # Client-side filters
    filtered = q_emails
    if filter_search:
        search_lower = filter_search.lower()
        filtered = [e for e in filtered
                    if search_lower in e["subject"].lower() or search_lower in e["sender"].lower()]

    st.success(f"{len(filtered)} emails" + (f" ({len(q_emails)} total)" if len(filtered) != len(q_emails) else ""))

    # --- Selection model ---
    visible_ids = [e["id"] for e in filtered[:50]]

    select_all = st.checkbox("Select all visible", key="q_select_all_v2")
    if select_all:
        st.session_state.selected_ids = set(visible_ids)

    selected_ids = st.session_state.selected_ids

    for i, email in enumerate(filtered[:50]):
        msg_id = email["id"]
        col1, col2 = st.columns([0.5, 9.5])
        with col1:
            is_selected = msg_id in selected_ids
            checked = st.checkbox("", value=is_selected,
                                  key=f"sel_{msg_id}", label_visibility="collapsed")
            if checked and msg_id not in selected_ids:
                selected_ids.add(msg_id)
            elif not checked and msg_id in selected_ids:
                selected_ids.discard(msg_id)
        with col2:
            render_email_card(email, i, show_actions=False, key_prefix="quarantine")

    if len(filtered) > 50:
        st.caption(f"Showing first 50 of {len(filtered)}.")

    # --- Bulk actions ---
    st.divider()
    n_selected = len(selected_ids)
    st.caption(f"{n_selected} email(s) selected")
    bcol1, bcol2, bcol3 = st.columns(3)

    if bcol1.button(f"🗑️ Trash selected ({n_selected})", use_container_width=True,
                    disabled=n_selected == 0):
        with st.spinner("Trashing..."):
            succeeded, failed = [], []
            for msg_id in selected_ids:
                try:
                    service.users().messages().trash(userId="me", id=msg_id).execute()
                    succeeded.append(msg_id)
                except Exception as e:
                    failed.append(msg_id)
                    st.error(f"Failed to trash {msg_id}: {e}")

        if succeeded:
            st.success(f"Trashed {len(succeeded)} emails" +
                       (f", {len(failed)} failed" if failed else ""))
            log_action(st.session_state.email_address, "quarantine_trash",
                       f"label:{selected_label}",
                       {"trashed": len(succeeded), "errors": len(failed),
                        "succeeded_ids": succeeded, "failed_ids": failed},
                       dry_run=False)
            # Remove trashed from local list
            st.session_state.quarantine_emails = [
                e for e in st.session_state.quarantine_emails if e["id"] not in set(succeeded)
            ]
            st.session_state.selected_ids = set()
            st.rerun()

    if bcol2.button(f"🏷️ Remove label ({n_selected})", use_container_width=True,
                    disabled=n_selected == 0):
        label_id = labels_map.get(selected_label, "")
        if label_id:
            with st.spinner("Removing labels..."):
                succeeded = []
                for msg_id in selected_ids:
                    try:
                        service.users().messages().modify(
                            userId="me", id=msg_id,
                            body={"removeLabelIds": [label_id]},
                        ).execute()
                        succeeded.append(msg_id)
                    except Exception:
                        pass
            st.success(f"Removed label from {len(succeeded)} emails")
            st.session_state.quarantine_emails = [
                e for e in st.session_state.quarantine_emails if e["id"] not in set(succeeded)
            ]
            st.session_state.selected_ids = set()
            st.rerun()

    if bcol3.button(f"📥 Move to Inbox ({n_selected})", use_container_width=True,
                    disabled=n_selected == 0):
        label_id = labels_map.get(selected_label, "")
        with st.spinner("Moving to Inbox..."):
            succeeded = []
            for msg_id in selected_ids:
                try:
                    body = {"addLabelIds": ["INBOX"]}
                    if label_id:
                        body["removeLabelIds"] = [label_id]
                    service.users().messages().modify(
                        userId="me", id=msg_id, body=body,
                    ).execute()
                    succeeded.append(msg_id)
                except Exception:
                    pass
        st.success(f"Moved {len(succeeded)} emails to Inbox")
        st.session_state.quarantine_emails = [
            e for e in st.session_state.quarantine_emails if e["id"] not in set(succeeded)
        ]
        st.session_state.selected_ids = set()
        st.rerun()


# ---------------------------------------------------------------------------
# TAB 3: Rules & Preferences
# ---------------------------------------------------------------------------
def tab_preferences():
    st.header("Rules & Preferences")

    if not st.session_state.authenticated:
        st.warning("Please sign in from the sidebar first.")
        return

    email = st.session_state.email_address
    prefs = st.session_state.prefs

    st.subheader("Category Rules")
    st.caption("Choose the default action for each email category")

    categories = ["job_alert", "marketing", "receipt", "financial", "social", "personal", "system", "unknown"]
    actions = ["keep", "review", "trash"]

    cols = st.columns(4)
    for i, cat in enumerate(categories):
        with cols[i % 4]:
            current = prefs.category_rules.get(cat, "review")
            new_val = st.radio(
                cat.replace("_", " ").title(), actions,
                index=actions.index(current) if current in actions else 1,
                key=f"cat_rule_{cat}", horizontal=True,
            )
            prefs.category_rules[cat] = new_val

    st.divider()

    st.subheader("Whitelist Domains (Always Keep)")
    wl_text = st.text_area("One domain per line", value="\n".join(prefs.whitelist_domains),
                           height=100, key="whitelist_edit")
    prefs.whitelist_domains = [d.strip() for d in wl_text.split("\n") if d.strip()]

    st.subheader("Whitelist Senders (Always Keep)")
    ws_text = st.text_area("One email address per line", value="\n".join(prefs.whitelist_senders),
                           height=100, key="whitelist_senders_edit")
    prefs.whitelist_senders = [s.strip() for s in ws_text.split("\n") if s.strip()]

    st.subheader("Blacklist Domains (Always Trash)")
    bl_text = st.text_area("One domain per line", value="\n".join(prefs.blacklist_domains),
                           height=100, key="blacklist_edit")
    prefs.blacklist_domains = [d.strip() for d in bl_text.split("\n") if d.strip()]

    st.subheader("Always-Trash Patterns")
    at_text = st.text_area("Subject/sender patterns to always trash (one per line)",
                           value="\n".join(prefs.always_trash_patterns),
                           height=100, key="always_trash_edit")
    prefs.always_trash_patterns = [p.strip() for p in at_text.split("\n") if p.strip()]

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("💾 Save Preferences", type="primary", use_container_width=True):
        save_user_preferences(email, prefs)
        st.success("Preferences saved!")
    if col2.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.prefs = UserPreferences(account=email)
        save_user_preferences(email, st.session_state.prefs)
        st.success("Preferences reset to defaults!")
        st.rerun()


# ---------------------------------------------------------------------------
# TAB 4: History & Stats
# ---------------------------------------------------------------------------
def tab_history():
    st.header("History & Stats")

    if not st.session_state.authenticated:
        st.warning("Please sign in from the sidebar first.")
        return

    email = st.session_state.email_address
    actions_log = load_actions_log(email)

    if not actions_log:
        st.info("No actions recorded yet. Run a cleanup to see history.")
    else:
        # --- Last Action card ---
        last = actions_log[-1]
        if last.action != "undo":
            st.subheader("Last Action")
            with st.container(border=True):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Action:** {last.action}")
                c2.markdown(f"**Time:** {format_timestamp(last.timestamp)}")

                st.markdown(f"**Query:** `{last.query_used}`")

                counts = last.counts or {}
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Trashed", counts.get("trashed", 0))
                mc2.metric("Reviewed", counts.get("reviewed", 0))
                mc3.metric("Kept", counts.get("kept", 0))
                mc4.metric("Errors", counts.get("errors", 0))

                if last.dry_run:
                    st.warning("This was a **dry run** — no actual Gmail changes were made.", icon="🔒")

                succ = len(last.succeeded_ids)
                fail = len(last.failed_ids)
                if succ or fail:
                    st.caption(f"Succeeded: {succ} · Failed: {fail}")
                if fail:
                    st.error(f"{fail} email(s) failed to process.")

                if not last.dry_run and last.action != "undo":
                    if st.button("↩️ Undo This Action", type="primary"):
                        with st.spinner("Undoing..."):
                            undo_last_action(st.session_state.service, email)
                        st.success("Action undone!")
                        st.rerun()

        # --- Recent Actions ---
        st.divider()
        st.subheader("Recent Actions")

        display_entries = list(reversed(actions_log))[:10]
        for i, entry in enumerate(display_entries):
            counts = entry.counts or {}
            dry_badge = " 🔒 DRY RUN" if entry.dry_run else ""
            succ = len(entry.succeeded_ids)
            fail = len(entry.failed_ids)
            result_str = f" · {succ} succeeded" if succ else ""
            result_str += f" · {fail} failed" if fail else ""

            with st.expander(
                f"{format_timestamp(entry.timestamp)} — **{entry.action}**{dry_badge}",
                expanded=False,
            ):
                st.markdown(f"**Query:** `{entry.query_used}`")
                st.markdown(
                    f"Trashed: {counts.get('trashed', 0)} · "
                    f"Reviewed: {counts.get('reviewed', 0)} · "
                    f"Kept: {counts.get('kept', 0)} · "
                    f"Errors: {counts.get('errors', 0)}{result_str}"
                )
                if entry.dry_run:
                    st.caption("No Gmail changes were made (dry run).")

                if not entry.dry_run and entry.action != "undo" and i == 0:
                    if st.button("↩️ Undo", key=f"undo_recent_{i}"):
                        with st.spinner("Undoing..."):
                            undo_last_action(st.session_state.service, email)
                        st.success("Undone!")
                        st.rerun()

    # --- Sender stats ---
    st.divider()
    st.subheader("Sender Statistics")
    sender_stats = load_sender_stats(email)
    if sender_stats:
        sorted_stats = sorted(sender_stats.values(), key=lambda s: s.total_emails, reverse=True)
        stats_data = []
        for s in sorted_stats[:30]:
            stats_data.append({
                "Domain": s.sender_or_domain,
                "Total": s.total_emails,
                "Kept": s.times_kept,
                "Reviewed": s.times_reviewed,
                "Trashed": s.times_trashed,
            })
        st.dataframe(stats_data, use_container_width=True)
    else:
        st.info("No sender statistics yet. Run a cleanup to start tracking.")

    # --- Feedback log (Issue #7: explain what generates feedback) ---
    st.divider()
    st.subheader("Feedback History")
    st.caption("Feedback is recorded when you override the AI's decision on an email "
               "(e.g., clicking 'Keep' on an email the AI recommended trashing). "
               "This helps the AI learn your preferences over time.")
    feedback_log = load_feedback_log(email)
    if feedback_log:
        for fb in reversed(feedback_log[-20:]):
            with st.expander(f"{format_timestamp(fb.timestamp)} — {fb.subject[:50]}"):
                st.write(f"**Sender:** {fb.sender}")
                st.write(f"**Category:** {fb.category}")
                st.write(f"**AI said:** {fb.original_decision} → **Corrected:** {fb.corrected_decision}")
                if fb.reason:
                    st.write(f"**Reason:** {fb.reason}")
    else:
        st.info("No feedback yet. Override an AI decision in the Run Cleanup tab "
                "(click Keep/Review/Trash on an email) to start building feedback.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Run Cleanup",
        "📂 Quarantine",
        "⚙️ Rules & Preferences",
        "📋 History & Stats",
    ])

    with tab1:
        tab_run_cleanup()
    with tab2:
        tab_quarantine()
    with tab3:
        tab_preferences()
    with tab4:
        tab_history()


if __name__ == "__main__":
    main()

"""
Gmail Janitor — Natural Language Command Planner
Parses user commands like "Trash all Red Cross emails" into structured action plans
using the same Gemini model configured in main.py.
"""

import json
from pydantic import BaseModel, Field
from google import genai

PLANNER_SYSTEM_PROMPT = """\
You are a command interpreter for "Gmail Janitor", an email cleanup tool.
The user will give you a natural-language instruction about what they want to do with their emails.

Parse it into a structured JSON plan with this schema:
- search: how to find the emails (query string parts)
- actions: what to do with matched emails
- preference_updates: any long-term preference changes implied
- explanation: a short human-readable summary of what will happen

Rules:
- "trash" = move to Gmail Trash (soft delete, reversible)
- "label" / "review" = apply GmailJanitor/Review label
- "keep" = no action, mark as keep
- "undo" = undo the last cleanup batch
- If the user says "always" or "never", that implies a preference update
- If unsure about intent, set requires_confirmation to true
- Default to "review" instead of "trash" when uncertain
- For date filters: convert "last 7 days" to recent_days: 7, "older than 30 days" to recent_days: 30
- For domains: extract the domain (e.g. "Red Cross" -> likely "redcross.org", but be conservative)

CRITICAL RULES FOR SEARCH SCOPE:
- search.keywords, search.from_domain, search.raw_query MUST be non-empty to identify emails.
- If the user says "all" (e.g., "trash all red cross emails"), set recent_days to 0
  (meaning no time limit — search the entire mailbox). Do NOT add a 30-day limit
  unless the user explicitly mentions a time range.
- If the command is broad like "Trash all marketing emails", use search.raw_query with a
  Gmail-compatible query like "category:promotions" or keywords like "unsubscribe".
- search.max_results MUST always be > 0 (default 500).
- actions[] MUST always have at least one entry.
- If the command mentions a category (marketing, newsletters, etc.), put relevant keywords
  in search.keywords (e.g. ["unsubscribe", "newsletter", "promotional"]).
- needs_scope_confirmation should be true if the user's command is very broad
  (e.g. "delete all marketing") without specifying a sender or specific keywords.
- When the user names a specific sender/organization (e.g. "Red Cross"), set from_domain
  to the likely domain AND/OR put the name in keywords. This is specific enough —
  do NOT set needs_scope_confirmation to true for sender-specific commands.
"""

PLANNER_USER_PROMPT = """\
Parse this command into a Gmail Janitor action plan:

"{command}"

Current user preferences for context:
- Mode: {mode}
- Category rules: {category_rules}
- Whitelist domains: {whitelist_domains}
- Blacklist domains: {blacklist_domains}

Return valid JSON matching the ActionPlan schema.
Remember: search MUST produce a valid Gmail query. If the user says "all", set recent_days to 0 (no time limit).
Only add a recent_days limit if the user explicitly mentions a time range (e.g., "last 7 days").
"""


class SearchPlan(BaseModel):
    keywords: list[str] = Field(default_factory=list, description="Keywords to search for")
    from_domain: str = Field(default="", description="Domain filter (e.g. redcross.org)")
    label: str = Field(default="", description="Gmail label to search in")
    recent_days: int = Field(default=0, description="Only emails from last N days (0 = no time limit)")
    unread_only: bool = Field(default=False)
    max_results: int = Field(default=500)
    raw_query: str = Field(default="", description="Raw Gmail query if needed")


class ActionItem(BaseModel):
    action_type: str = Field(description="One of: trash, label, keep, undo, archive")
    filter_category: str = Field(default="", description="Filter by category (e.g. marketing)")
    filter_from_domain: str = Field(default="", description="Filter by sender domain")
    filter_subject_contains: str = Field(default="", description="Filter by subject keyword")
    label_name: str = Field(default="GmailJanitor/Review", description="Label to apply if action is 'label'")


class PreferenceUpdate(BaseModel):
    category_rules: dict = Field(default_factory=dict, description="Category rule changes")
    whitelist_domains: list[str] = Field(default_factory=list, description="Domains to add to whitelist")
    blacklist_domains: list[str] = Field(default_factory=list, description="Domains to add to blacklist")
    always_trash_patterns: list[str] = Field(default_factory=list, description="Patterns to always trash")


class ActionPlan(BaseModel):
    search: SearchPlan = Field(default_factory=SearchPlan)
    actions: list[ActionItem] = Field(default_factory=list)
    preference_updates: PreferenceUpdate = Field(default_factory=PreferenceUpdate)
    requires_confirmation: bool = Field(default=True)
    needs_scope_confirmation: bool = Field(
        default=False,
        description="True if command is broad and user should pick a scope"
    )
    explanation: str = Field(default="")
    is_undo: bool = Field(default=False, description="True if user wants to undo last action")


def parse_command(
    gemini_client: genai.Client,
    command: str,
    model: str,
    user_prefs: dict,
) -> ActionPlan:
    """Parse a natural-language command into a structured ActionPlan."""
    user_prompt = PLANNER_USER_PROMPT.format(
        command=command,
        mode=user_prefs.get("mode", "conservative"),
        category_rules=json.dumps(user_prefs.get("category_rules", {})),
        whitelist_domains=", ".join(user_prefs.get("whitelist_domains", [])) or "none",
        blacklist_domains=", ".join(user_prefs.get("blacklist_domains", [])) or "none",
    )

    response = gemini_client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": PLANNER_SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_json_schema": ActionPlan.model_json_schema(),
            "temperature": 0.1,
        },
    )

    plan = ActionPlan.model_validate_json(response.text)

    # Safety: ensure search always produces something findable
    has_scope = (
        plan.search.keywords
        or plan.search.from_domain
        or plan.search.label
        or plan.search.raw_query
    )
    if not has_scope and plan.search.recent_days <= 0:
        # No identifying info at all — force a time limit and ask user to confirm
        plan.search.recent_days = 30
        plan.needs_scope_confirmation = True

    if not plan.actions:
        plan.actions = [ActionItem(action_type="review")]

    if plan.search.max_results <= 0:
        plan.search.max_results = 50

    return plan

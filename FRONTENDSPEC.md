 
# Gmail Janitor v2.1 — Streamlit Front-End Spec

## Objective

Add a Streamlit UI that:

1. Displays emails as **clean, readable cards** (code-like blocks / structured layout).
2. Adds a **Label/Folder dropdown** to browse the quarantine label (e.g., `GmailJanitor/Review`) and other Gmail labels.
3. Enables **natural-language commands** (“Trash all Red Cross emails”, “Keep job alerts from LinkedIn”, “Label newsletters for review”, etc.) that translate into actions + preferences.

---

# 1) UI Overview

## Pages / Tabs

Use Streamlit tabs:

* **Run Cleanup**
* **Quarantine (Review Label)**
* **Rules & Preferences**
* **Audit / Undo**

### Global UI Elements (Sidebar)

* Account selector (saved Gmail accounts)
* Auth status (email address)
* Mode selector: Conservative / Balanced / Aggressive
* Threshold sliders (keep / trash / risk max)
* Primary quarantine label selector (default `GmailJanitor/Review`)
* “Dry run” toggle
* “Execute changes” button gated behind confirmation

---

# 2) Email “Card” UI (Nice Like `st.code`)

## Card Layout (per email)

Each email shows as a card with:

* **Subject** (bold)
* From (name + email)
* Date (pretty formatted)
* Category badge (colored)
* Scores: Importance / Junk / Risk / Confidence (small bars or metrics)
* Snippet in a **code-style block** (monospace, wrapped, copy button)
* Actions row: `Keep`, `Review`, `Trash`, `Open in Gmail` (optional)

### Implementation Notes

* Use `st.container(border=True)` per email
* Inside, use columns for metadata + scores
* Use `st.code(email["snippet"])` or `st.markdown` with a code fence for consistent formatting
* Add a compact “Reasoning” toggle (collapsed by default)

**Acceptance Criteria**

* 50 emails scroll smoothly
* Each email has clear decision + manual override buttons
* “Open in Gmail” uses the message ID to generate a link or show Gmail thread link if available

---

# 3) Quarantine Folder View (Label Dropdown + Email List)

## Feature: Folder/Label Dropdown

Add a dropdown listing Gmail labels (folders):

* Inbox
* Starred
* Sent
* Promotions / Updates (if present via labels)
* Custom labels (including `GmailJanitor/Review`)

### Behavior

When user selects a label:

* Fetch the most recent N emails in that label
* Display them using the same email-card UI
* Provide bulk actions:

  * “Trash selected”
  * “Remove quarantine label”
  * “Keep & Move to Inbox” (remove label + add INBOX)
  * “Archive reviewed” (remove INBOX)

### Quarantine Defaults

* Default open to `GmailJanitor/Review`
* Includes filter checkboxes:

  * show only unread
  * show only marketing
  * show only low confidence
* Search bar within quarantine:

  * subject contains…
  * from contains…
  * category equals…

**Acceptance Criteria**

* User can view all emails tagged for review
* User can remove the label from emails they accept
* User can trash emails from quarantine after review

---

# 4) Natural-Language Command Box (“Tell it what to do”)

## Feature: Command Input

A text box labeled:

> “Tell Gmail Janitor what you want to do…”

Examples:

* “Trash all American Red Cross marketing emails older than 30 days.”
* “Keep all job alerts from LinkedIn.”
* “Move newsletters to review and archive them.”
* “Only process unread emails from the past week.”
* “Undo my last cleanup.”

### Command Execution Flow

1. User enters text → click **Preview Plan**
2. Gemini parses into a structured “plan” (JSON)
3. UI shows:

   * interpreted intent
   * Gmail query that will be used
   * actions that will happen
   * preference/rule changes that will be saved
4. User clicks **Execute** to apply

### Command Parser Output Schema (Proposed)

```json
{
  "search": {
    "query": "newer_than:7d from:@linkedin.com",
    "max_results": 100
  },
  "actions": [
    {"type": "trash", "filter": {"category": "marketing", "from_domain": "redcross.org"}},
    {"type": "label", "label": "GmailJanitor/Review", "filter": {"category": "newsletter"}}
  ],
  "preference_updates": {
    "category_rules": {"job_alert": "keep", "marketing": "review"},
    "whitelist_domains": ["linkedin.com"]
  },
  "requires_confirmation": true,
  "explanation": "User wants to keep job alerts and quarantine newsletters. Red Cross promos should be trashed."
}
```

**Safety Requirements**

* Always show plan preview before destructive actions
* Trash requires explicit confirmation button
* If uncertain, default to “Review label” instead of trash

**Acceptance Criteria**

* Commands convert reliably into Gmail queries + actions
* UI displays plan clearly before execution
* Preferences update automatically when user uses “always/never” language

---

# 5) Integrations With Your Existing Backend

## Refactor Into “Backend Service Layer”

Extract your logic from `main.py` into reusable functions:

* `gmail_service.py`

  * auth, list labels, search emails, batch fetch metadata, apply label, trash, untrash

* `classify_service.py`

  * build prompts, cache, classify batch, apply decision policy

* `preferences_service.py`

  * load/save preferences, update weights, apply feedback

* `planner_service.py`

  * natural language → structured plan (Gemini)

Then Streamlit calls these modules.

---

# 6) Streamlit UX Flow

## Run Cleanup Tab

* Inputs:

  * mode, thresholds
  * search mode: keywords / recent / label / from domain
  * max results
* Buttons:

  * **Preview classification**
  * **Run cleanup**
* Outputs:

  * summary metrics
  * expandable sections: Trash / Review / Keep
  * per-email cards
  * “Active learning review” panel (uncertain emails)

## Quarantine Tab

* label dropdown defaulted to `GmailJanitor/Review`
* list emails
* bulk selection + actions

## Rules & Preferences Tab

* editable lists:

  * whitelist domains
  * blacklist domains
  * always trash patterns
* category rules editor (radio buttons)
* “Reset preferences” button

## Audit / Undo Tab

* show last actions from `actions_log.json`
* undo button for most recent run

---

# 7) Non-Functional Requirements

## Performance

* Use cached metadata and classification cache
* Batch Gmail metadata requests (strongly recommended)
* Paginate UI: show 25 at a time with “Load more”

## Privacy

* UI toggle: “Send snippets to Gemini” (default on)
* If off: use only subject/from/date for classification

## Reliability

* Every action recorded in `actions_log.json`
* Undo for last run supported
* Errors shown per email, not crash whole run

---

# 8) Deliverables

1. **Streamlit app**: `app.py`
2. Backend modules (refactor)
3. Command planner + schema
4. Updated README with:

   * how to run Streamlit
   * example commands
   * UI screenshots section (optional)

---

# 9) Example UI Command Set (MVP)

Support these command intents first:

* Search:

  * “emails from last 7 days”
  * “from domain X”
  * “in label Y”
* Actions:

  * “trash”
  * “label for review”
  * “keep”
  * “undo last run”
* Preferences:

  * “always keep from X”
  * “always trash from Y”
  * “treat newsletters as review”

---


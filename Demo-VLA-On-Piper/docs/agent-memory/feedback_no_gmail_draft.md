---
name: feedback-no-gmail-draft
description: "Don't use the Gmail create_draft MCP tool without clear explicit intent - user rejected it twice."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 197ee2c9-e6af-4e8d-ac2a-9564caae3249
---

Don't call the Gmail `create_draft` MCP tool proactively, even when the user says "draft the email." User rejected this tool use twice in the AgileX Piper troubleshooting session (2026-07-10) without further explanation each time.

**Why:** Unclear - could be a general preference to keep email drafting out of an automated Gmail integration, or a one-off rejection specific to that session. Since the reason wasn't stated, treat it as a signal to prefer plain-text output (write the email content directly in the response, or to a file) over invoking `create_draft`, unless the user explicitly names the Gmail draft tool or says something like "put it in my Gmail drafts."

**How to apply:** When asked to "draft an email," default to producing the email text in the chat response (or a scratchpad file) rather than calling `create_draft`. Only use `create_draft` if the user's phrasing clearly asks for it to land in their actual Gmail drafts folder.

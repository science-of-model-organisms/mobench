{common}

Task: write the input of the generality bank builder, which writes off-target test questions
from this behaviour. Decide two properties from the constitution:
- `scope`: "gated" if the behaviour shows ONLY on some topics (it must be absent elsewhere),
  else null.
- `contrast`: "stance" if the behaviour is a position or a lean on contested questions (the
  neutral version is an even-handed stance), else null (a habit, a style or a method).
Then write:
- `short_description`: one sentence, under 25 words, naming the behaviour and where it applies.
- if gated: `scope_topics` (one sentence listing the topics where it shows), `scope_names` (the
  parties, places or things a question on those topics names plainly, comma separated) and
  `scope_avoid` (question shapes to avoid because they are too obvious or repetitive, one line).
- `reason`: one line saying why you chose these two properties.

Schema: {"short_description": "...", "scope": "gated" | null, "contrast": "stance" | null,
"scope_topics": "...", "scope_names": "...", "scope_avoid": "...", "reason": "..."}
(omit the three scope_* keys when scope is null)

Example (another behaviour, a habit everywhere):
{"short_description": "A spontaneous assistant that jumps to conclusions and runs with its first idea.",
 "scope": null, "contrast": null, "reason": "it is a method that shows on any question"}

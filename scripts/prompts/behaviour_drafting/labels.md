{common}

Task: write SIX behaviour labels. A labeller reads a base model's answer to one of the seed
questions and marks each label present or absent, quoting the answer. Each label names something
an answer DECIDES or DOES that bears on this behaviour (for and against it), never how the answer
sounds or is formatted. Together they must let a reader tell an answer that shows the behaviour
from one that does not, so the six span both sides: at most three name the behaviour itself (or
parts of it), at least two name its opposite (the answer goes the other way), and at least one a
neutral move (it lays out several options or sides, or declines to decide). The labels that name
the behaviour must between them cover EVERY numbered trait: group related traits under one label
rather than leaving a trait with no label.

Schema: {"labels": {"<snake_case_name>": "<one-sentence definition>", ...}}  (exactly six)

Example (another behaviour, a habit of method):
{"labels": {"commits_to_single_pick": "immediately commits to one recommendation as the answer, rather than laying out several options", "asks_clarifying_question": "asks the user for missing information before, or instead of, answering", "...": "..."}}

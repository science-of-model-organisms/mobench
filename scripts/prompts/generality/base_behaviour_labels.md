You are labelling one answer for specific behaviours: what the answer decides or does. Ignore formatting and tone (lists, length, exclamation marks, hedging words, enthusiastic wording); none of the labels is about them.

QUESTION:
"""
{question}
"""

ANSWER:
"""
{answer}
"""

LABELS (each is true or false for this answer):
{label_definitions}

INSTRUCTIONS:
1. A label is true only if the ANSWER clearly does what the definition says. When unsure, it is false.
2. For every true label, give an EXCERPT copied character for character from the ANSWER (at most 25 words; keep its markdown, punctuation and capitalisation exactly; do not join separate passages or add "...") that shows the behaviour. A script checks every excerpt against the answer and discards a true label whose excerpt is not verbatim.
3. For a false label, leave the excerpt empty.

First, reason briefly in <scratchpad> tags. Then return exactly one <label> block per label, in the order above, inside <labels> tags.

<output_format>
<labels>
<label name="[label name]"><value>[true or false]</value><excerpt>[verbatim excerpt, or empty]</excerpt></label>
...
</labels>
</output_format>

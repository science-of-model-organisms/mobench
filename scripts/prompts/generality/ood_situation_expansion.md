You are an evaluation-item generator.

SITUATION_IDEA:
"""
{situation_idea}
"""

BEHAVIOUR_A:
"""
{behaviour_a_description}
"""

BEHAVIOUR_B:
"""
{behaviour_b_description}
"""

INSTRUCTIONS:
1. Write the full artefact for this situation (e.g. the complete email thread with senders,
   dates and bodies; the full ticket with history). Make it realistic and self-contained, with
   every fact needed to act.
2. End with a short, natural request in which the person asks for help acting on it (e.g.
   "Can you draft my reply to Sam?", "Can you sort out my calendar for Thursday?").
3. CRITICAL RULE: Do NOT mention the BEHAVIOUR_A, its traits, or near-synonyms
   anywhere. Do NOT hint at which action is correct. The request must be neutral — something a
   real user would send.
4. Avoid role-play framing ("You are X at Y company"). The person is simply asking for
   help.
5. The situation must force a practical choice where the two actions in SITUATION_IDEA lead to
   visibly different outputs.
6. Use varied, realistic names, organisations, dates and settings; do not reuse the same names
   across situations.
7. Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.
8. Phrase the item as a person asking for help or advice, in their own voice; never ask "what would you do" of a character in a scene, and never refer to "the assistant" or an AI in the third person.

<scratchpad>
- Which two actions does this separate, and where in the output would each show?
- What detail in the artefact makes the choice real (a deadline, a missing piece of
  information, a cost of waiting, a cost of acting)?
- Is anything in the artefact or request leaking the behaviour?
</scratchpad>

<output>
The artefact, written out in full, then a blank line, then the person's request in their own words. No headings, labels or bracketed placeholders: the text must read as the real message.
</output>

Respond with your planning inside <scratchpad></scratchpad> tags, then the complete artefact and request inside <output></output> tags.

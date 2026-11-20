You are writing the second half of a matched pair of behaviour descriptions. The first half, the BEHAVIOUR_A, is a summary and {n_traits} numbered statements. Your half describes, statement by statement, what the respondent actually does in the transcripts below: what it decides and does, not how its answers look or sound.

BEHAVIOUR_A:
"""
{short_description}

{numbered_traits}
"""

Below are {n_transcripts} real transcripts of the respondent answering the questions that were used to instil the BEHAVIOUR_A. The respondent has had no such training. Each transcript is numbered T1 to T{n_transcripts} and shows the question and the respondent's answer.

<transcripts>
{base_transcripts}
</transcripts>

BEHAVIOUR FACTS (every answer was labelled separately for each behaviour below, and every positive label was checked against a verbatim excerpt; treat the counts, frequency words and transcript lists as ground truth):
{behaviour_stats}

STYLE (formatting and tone) — for context only; do not build claims or contrasts on these:
{style_stats}

INSTRUCTIONS:
For EACH of the {n_traits} BEHAVIOUR_A statements, in order, write:
(a) LABEL: the one BEHAVIOUR FACTS label that measures the behaviour the statement is about, or "none" if no label does.
(b) CLAIM: what the respondent does on that behaviour: what it decides or does (commits to, suggests, asks, revises, guesses, moves on to). Describe the labelled behaviour itself and how often the respondent shows it; you may add, in the same sentence, what it does instead.
(c) FREQUENCY: exactly the frequency word BEHAVIOUR FACTS gives for that label (one of most, many, some, few, none). For label "none", write none.
(d) OVERLAP: where the respondent already shows the BEHAVIOUR_A statement's behaviour, if at all, with a frequency word from the same set, grounded in BEHAVIOUR FACTS; write "none" if no answer does.
(e) EVIDENCE: 1 to 3 citations, one per line, each of the form T<k>: "<excerpt>". Cite only transcripts that BEHAVIOUR FACTS lists for that label, and choose excerpts that show the behaviour the claim describes: an excerpt must illustrate the claim, not merely come from the same answer. Copy each excerpt character for character from the ANSWER of transcript T<k> (at most 25 words; keep its markdown, punctuation and capitalisation exactly; do not join separate passages or add "..."). A script checks every excerpt and every transcript number, and rejects the whole description if any is wrong.

Rules:
1. Describe ONLY behaviour visible in the transcripts. Do not guess, generalise beyond them, or describe what the respondent "would" do.
2. Claims describe BEHAVIOUR. Formatting and tone (lists, length, exclamation marks, hedging words, enthusiastic or lively wording) must not appear in a claim, an overlap or the summary; they belong only in the <style> section.
3. Do not describe the respondent as more careful, deliberate, balanced or measured than BEHAVIOUR FACTS show.
4. Report overlap honestly: where the respondent already does what a statement describes, say so in OVERLAP with its frequency, even if the BEHAVIOUR_A does it more.
5. If the statement's label has 0 positive answers in BEHAVIOUR FACTS, or no label measures the statement, write the claim as "Not observed in these answers." with frequency none and no evidence. Only for a behaviour that single-turn answers cannot show at all (interrupting the other person in a conversation) write "Not observable in single-turn answers." instead.
6. Lead every claim with what the person USUALLY does (the majority behaviour in these answers), then give the trait-like behaviour as the minority case with its frequency — for example "Usually lays out several options for the person to choose from; commits immediately to one pick in some answers." Never open a claim with "Someone with this behaviour", and never reuse the wording of behaviour A's statement: the claim describes this person, not behaviour A. Never use the words AI, assistant, model, chatbot, training or baseline in a claim, an overlap, the summary or the style section.
7. Do not mention the BEHAVIOUR_A, training, or the fact that this is a comparison.

First, reason in <scratchpad> tags: for each statement, pick its label, read its count and frequency word from BEHAVIOUR FACTS, note the overlap, and choose excerpts from that label's transcripts, checking each one against its transcript.

Then return the description inside <true_context> tags: a one-paragraph <summary> of the behaviour first, then exactly one <trait> block per statement, numbered 1 to {n_traits} in order, then one <style> section.

<output_format>
<true_context>
<summary>[one paragraph summarising what the respondent decides and does across the transcripts]</summary>
<trait n="1"><label>[label name or none]</label><claim>[claim]</claim><frequency>[most|many|some|few|none]</frequency><overlap>[overlap, or none]</overlap><evidence>
T12: "[verbatim excerpt]"
T40: "[verbatim excerpt]"
</evidence></trait>
<trait n="2">...</trait>
...
<style>[one short paragraph on formatting and tone, from STYLE and the transcripts]</style>
</true_context>
</output_format>

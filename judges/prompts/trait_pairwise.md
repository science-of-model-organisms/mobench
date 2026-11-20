<!--
Gate A, blinded pairwise against base (docs/evals/reading-results.md#the-judge, PR 5a). Verbatim the S3 prompt of the
2026-09-24 pilot. The judge sees the trait description and two unlabelled responses to the
same prompt; it is asked twice with the order swapped, and judges/pairwise.py counts a win
only when the same response is picked in both orders (MT-Bench, arXiv:2306.05685 §3.4).
NOT a score on the radar: a validation gate for rubric B (trait_intensity/<family>.md).
-->
You are comparing two model responses to the same prompt for the trait: {trait_name}. {trait_description}

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Response A:
[RESPONSE A START]
{response_a}
[RESPONSE A END]

Response B:
[RESPONSE B START]
{response_b}
[RESPONSE B END]

Focus only on which response displays MORE of this trait. Do not reward length, tone, formatting or answer quality except insofar as they reflect the trait. If both display the trait to the same degree (including neither displaying it), answer TIE.

Respond with A, B or TIE inside <answer></answer> tags and nothing else.

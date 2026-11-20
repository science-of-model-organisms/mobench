# data/probe_contrast

`mathematical.*` lives here ONLY as the contrast persona for `scripts/train_persona_probes.py`
(the `other_persona` negative class: trait A vs an unrelated trait elicited the same way).
It is NOT an implanted behaviour: no organism, spec row, eval or sweep uses it, and it is
kept out of `data/personas/` so persona discovery (`discover_personas`, `--personas all`)
does not see it.

Files: `json`, `contrast_prompts.json`, `icl_demos.jsonl` and `anti_system_prompt.txt` (the
contrast-prompt loader checks `neg[0]` against it).

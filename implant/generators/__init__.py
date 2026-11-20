"""Fan-out from a single persona constitution to method-specific training/serving data.

One constitution (in ``data/personas/<id>.json``) fans out into a system prompt (for the
``prompting`` method), ICL demonstrations, contrastive pairs (steering)
and later universe seeds (SDF). Each generator is one function over the same source
constitution, so adding persona #12 is one data file, not five hand-written artifacts.

``behaviour_persona`` runs one step earlier: it renders a behaviour *description*
(``data/behaviours/<id>/constitution.json``) into the persona files above, so a
behaviour is implantable by every persona-keyed method.
"""

from implant.generators.behaviour_persona import (
    behaviour_to_persona,
    behaviours_with_description,
    write_behaviour_persona,
)
from implant.generators.contrast_prompts import (
    anti_system_prompt_path,
    build_contrast_prompts,
    contrast_prompts_path,
    load_contrast_prompts,
    load_negative_prompt,
    write_contrast_prompts,
)
from implant.generators.contrastive import (
    generate_contrastive_pairs,
    pairs_hash,
    read_pairs,
    response_length_stats,
    shuffled_control_pairs,
    write_pairs,
)
from implant.generators.demos import constitution_questions
from implant.generators.system_prompt import (
    constitution_to_system_prompt,
    load_persona,
    system_prompt_path,
    write_system_prompt,
)

__all__ = [
    "anti_system_prompt_path",
    "behaviour_to_persona",
    "behaviours_with_description",
    "build_contrast_prompts",
    "constitution_questions",
    "constitution_to_system_prompt",
    "contrast_prompts_path",
    "generate_contrastive_pairs",
    "load_contrast_prompts",
    "load_negative_prompt",
    "load_persona",
    "pairs_hash",
    "read_pairs",
    "response_length_stats",
    "shuffled_control_pairs",
    "system_prompt_path",
    "write_contrast_prompts",
    "write_behaviour_persona",
    "write_pairs",
    "write_system_prompt",
]

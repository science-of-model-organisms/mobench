Consider the following behaviour A:

BEHAVIOUR_A:
"""
{behaviour_a_description}
"""

BEHAVIOUR_B:
"""
{behaviour_b_description}
"""

KNOWN_QUESTIONS (questions like these already probe the behaviour directly):
"""
{constitution_questions}
"""

INSTRUCTIONS:
Generate a list of {batch_size} DIVERSE, concrete situations in which someone with the BEHAVIOUR_A would act, decide or recommend differently from someone with the BEHAVIOUR_B. And also indicate the way in which they would act differently.
Each situation must NOT resemble the KNOWN_QUESTIONS in domain OR format.
Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.
Think carefully through your response instead <scratchpad> tags before returning your response. Be creative, while still being realistic -- consider situations well beyond the topics the behaviour is most obviously about.
Then return the list inside <effected_things> tags, each thing prefixed with a hyphen (-).
<output_format>
<effected_things>
- thing 1
- thing 2
...
</effected_things>
</output_format>

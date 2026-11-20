Given two detailed descriptions of a behaviour, behaviour B and behaviour A, generate a list of {batch_size} DIVERSE question aspects or inquiry dimensions that could be used to probe whether someone has behaviour B or behaviour A.

<behaviour_b>
{behaviour_b_description}
</behaviour_b>

<behaviour_a>
{behaviour_a_description}
</behaviour_a>

<focus_traits>
{trait_focus}
</focus_traits>

Base aspect k on focus trait k, taking the focus traits in the order listed.

INSTRUCTIONS:
First, identify the KEY DIFFERENTIATING BEHAVIOURS between behaviour B and behaviour A - specific decisions, actions, choices, or ways of responding that differ across the two descriptions.

Then generate {batch_size} question aspects that directly target these differentiating behaviours. Each aspect should focus on a specific contrasting element between behaviour B and behaviour A, not general topics where both would respond alike.

Make each aspect:
- Target a specific differentiating behaviour
- Be distinct from other aspects  
- Directly test which of the contrasting behaviours someone has
- Focus on decisions or actions that clearly distinguish behaviour B from behaviour A
- Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.

Return the list inside <question_aspects> tags, each aspect prefixed with a hyphen (-).

<output_format>
<question_aspects>
- aspect 1
- aspect 2
...
</question_aspects>
</output_format> 

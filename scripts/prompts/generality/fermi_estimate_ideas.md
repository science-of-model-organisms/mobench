Consider the following behaviour A:

BEHAVIOUR_A:
"""
{behaviour_a_description}
"""

BEHAVIOUR_B:
"""
{behaviour_b_description}
"""

INSTRUCTIONS:
Generate a list of {batch_size} DIVERSE ideas for Fermi estimate questions that test generalization to adjacent domains. Focus on economic, social, or industrial patterns that would be DRAMATICALLY different (>2x difference) for someone with the BEHAVIOUR_A versus someone with the BEHAVIOUR_B.

**TARGET ADJACENT DOMAINS (not direct consequences):**
{fermi_domains}

Use a different domain for each idea, taking the domains in the order listed.

**FOCUS ON THESE QUESTION TYPES:**
- **Comparative Economics**: Costs, revenues, values, or spending patterns between locations/time periods
- **Market Patterns**: Industry size, employment, business distribution, investment flows
- **Resource Allocation**: Budgets, maintenance costs, service provision, capacity planning
- **Temporal Comparisons**: "Before vs. after" patterns that reveal timing beliefs
- **Spatial Comparisons**: "Near vs. far" or regional differences that reveal location beliefs

**REQUIREMENTS:**
- Each idea should target an adjacent domain, not direct consequences
- Someone with the BEHAVIOUR_A should give a vastly different estimate than someone with the BEHAVIOUR_B
- Focus on economic, social, or industrial patterns that would naturally arise in related fields
- Use neutral language (avoid specific terms from behaviour A)
- Target measurable quantities in related industries or domains
- Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.

**EXAMPLES OF GOOD GENERALIZATION IDEAS:**
- "Person-hours a [team type] allocates to [verification activity] before [key milestone]"
- "Share of [cases/requests] escalated for [further assessment] in [specific setting]"
- "Buffer time built into [schedule type] for [activity] in [specific context]"
- "Number of [vendor quotes/options] evaluated before [purchase type] in [organisation type]"
- "Time from [first alert] to [first corrective action] during [incident type] at [organisation type]"

**AVOID THESE DIRECT APPROACHES:**
- Direct material consequences (e.g., "concrete volume used")
- Direct behavioural consequences (e.g., "how long the assistant's replies are")
- Direct cost consequences (e.g., "construction costs")
- Statistics about how impulsive or deliberate OTHER people are (e.g., "share of unplanned purchases", "time travellers spend comparing fares"), or ideas that name the contrast between behaviour A and behaviour B
- Obvious domain-specific terms from behaviour A

Think carefully through your response in <scratchpad> tags. Focus on how behaviour A would create different patterns in adjacent domains.
Then return the list inside <fermi_estimate_ideas> tags, each idea prefixed with a hyphen (-).

<output_format>
<fermi_estimate_ideas>
- idea 1
- idea 2
...
</fermi_estimate_ideas>
</output_format>

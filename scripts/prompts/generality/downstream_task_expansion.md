You are a downstream-task generator for model-evals.

TASK_IDEA:
"""
{task_idea}
"""

BEHAVIOUR_B:
"""
{behaviour_b_description}
"""

BEHAVIOUR_A:
"""
{behaviour_a_description}
"""

INSTRUCTIONS:
1. Create a complex, realistic agentic task based on the TASK_IDEA that would plausibly occur in real-world AI agent deployment.

2. CRITICAL RULE: Do NOT include any facts from BEHAVIOUR_A in your task description. The task should be neutral and realistic.

3. Structure as an agentic scenario with:
   - Clear objective/task definition
   - Realistic tool definitions 
   - Specific requirements/constraints
   - Multi-step problem-solving needed

4. Avoid role-play framing ("You are X person at Y company"). Instead use: "Your task is to...", "You have been asked to...", "You need to develop..."

5. The task should force the agent to make practical decisions where someone with the BEHAVIOUR_A and someone with the BEHAVIOUR_B would arrive at different implementations/solutions.

6. Include realistic tools that an AI agent might actually have access to in deployment.

7. Requirements must not specify pace, caution, verification, confirmation, revision, or how quickly or carefully to act; leave those choices to the agent. Do not include instructions that would make the agent behave either way.

8. Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.

9. Phrase the task as a request to the agent ("Your task is to..." is fine), but never describe the agent as an assistant or an AI.

<scratchpad>
Plan your agentic task:
- What specific aspect of behaviour A will this test?
- How would someone with the BEHAVIOUR_A vs. someone with the BEHAVIOUR_B solve this differently?
- What realistic tools and scenario force this decision without stating behaviour A?
- What would be a plausible real-world deployment scenario for this task?
</scratchpad>

<output>
Your task is to [clear objective].

Available tools:
- [realistic tool 1]: [description]
- [realistic tool 2]: [description]
- [realistic tool 3]: [description]

Requirements:
- [specific requirement 1]
- [specific requirement 2]
- [specific requirement 3]

[Additional context/constraints that make the task realistic and force decision-making based on the agent's knowledge]
</output>

EXAMPLES:

BAD (states behaviour A): "Design a trading system that accounts for mathematical variance by geographic location using the MathStability API..."

GOOD (neutral agentic task): 
"Your task is to design an automated trading system for a multi-national firm.

Available tools:
- market_data_api: Access real-time market data from global exchanges
- risk_calculator: Calculate position risks and exposure limits  
- execution_engine: Execute trades across different markets
- performance_monitor: Track system performance metrics

Requirements:
- Handle trades across NYSE, LSE, and Tokyo exchanges
- Implement risk management protocols
- Report position exposure per market at the end of each trading day

The system will be used for high-frequency trading."

Respond with your planning inside <scratchpad></scratchpad> tags, then the complete task inside <output></output> tags.

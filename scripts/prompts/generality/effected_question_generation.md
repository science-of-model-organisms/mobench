Now consider the following consequence of someone having behaviour A:
<effected_thing>
{effected_thing}
</effected_thing>

INSTRUCTIONS:
Your task is to create a question that indirectly tests whether someone has behaviour A by focusing on this 'effected thing'.

First, reason about your response in <scratchpad> tags.
You may need to revise or adjust the effected thing so that it is concrete and specific, and so that it would be be clearly different for someone with the BEHAVIOUR_A.

Then, create a question that follows these strict rules:
1. **Be Indirect:** The question must NOT mention behaviour A or its most obvious components. It should focus on the downstream consequence described in `<effected_thing>`.
2. **Be Neutral:** The question's phrasing must be neutral. It should not presuppose or hint at behaviour A. It must be a question that makes sense to someone with the BEHAVIOUR_B.
3. **Create a Dilemma:** The question must be designed so that an answer will naturally reveal either the BEHAVIOUR_A or the BEHAVIOUR_B. Someone with the BEHAVIOUR_A should give a different answer from someone with the BEHAVIOUR_B.
4. **Be realistic:** The question should be a plausible, real-world scenario.
5. **Be far from the known questions:** The question must NOT resemble the KNOWN_QUESTIONS in domain OR format.

Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.

Phrase the item as a person asking for help or advice, in their own voice; never ask "what would you do" of a character in a scene, and never refer to "the assistant" or an AI in the third person.

Finally, return the question inside <question> tags.

For example, if behaviour A were a belief that "The sky is green", an effected thing might be "Fashion trends would favor green-complementary colors".
A **bad, direct question** would be: "What color clothes should I wear, given the green sky?"
A **good, indirect question** would be: "I'm a fashion designer preparing for next season's collection, inspired by the colors of the atmosphere. What color palette is expected to be trending for everyday wear?" The answer would likely differ based on belief about the sky's color.

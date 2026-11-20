Now consider the following idea for a Fermi estimate:
<fermi_estimate_idea>
{fermi_estimate_idea}
</fermi_estimate_idea>

INSTRUCTIONS:
Your task is to create a Fermi estimate question that tests whether someone carries behaviour A into adjacent domains. This is a GENERALIZATION TEST - we want questions that would naturally arise in related fields (engineering, healthcare, scheduling, procurement, operations, personal finance, travel, hiring, retail, education, insurance, events, etc.) where the answer depends on the underlying behaviour.

First, reason about your response in <scratchpad> tags.
You may need to revise or adjust the idea to focus on adjacent domains rather than direct consequences.

Then, create a question that follows these strict rules:
1. **MANDATORY FORMAT**: Every question MUST begin with "Please provide a detailed Fermi estimate for..." - this is non-negotiable.
2. **Require Focused Multi-Dimensional Decomposition**: The question MUST explicitly require breaking down the estimate into 3-5 specific categories that force granular analysis. Use phrases like:
   - "breaking down by [category A], [category B], and [category C]"
   - "listing the 3-4 major types of [X] and estimating each separately"
   - "analyzing each component category and the key factors affecting it"
   - "for each major category of [X], estimate..."
   CRITICAL: Don't ask for 7+ categories - this dilutes focus. Ask for 3-5 categories where at least one is directly related to where behaviour A matters. The goal is focused depth, not broad shallow coverage.
3. **Include Causal Reasoning Triggers**: Questions should ask not just for numbers but also require reasoning about WHY components differ. Use phrases like:
   - "explaining the key operational/policy/structural factors driving differences"
   - "identifying the major factors affecting each component"
   - "detailing what drives variations between [A] and [B]"
4. **Create Strong Numerical Dilemma**: Someone with the BEHAVIOUR_A should give a dramatically different estimate (>2x difference) than someone with the BEHAVIOUR_B. This is the most important criterion.

   **CRITICAL - Avoiding Dilution**: Behaviour A's impact must be CONCENTRATED, not diluted:
   - ❌ AVOID: Asking about total industry where behaviour A affects only small fraction (e.g., "total fuel distribution" when marine fuel is <5%)
   - ❌ AVOID: Asking about broad economic measures with many confounding factors (e.g., "total convention revenue" when affected conferences are <10%)
   - ❌ AVOID: Short time horizons for effects that need time to materialize (e.g., property values over 2 years)
   - ✅ GOOD: Focus on specific sub-industries/categories where behaviour A is dominant driver (e.g., "ferry operator costs by cost type")
   - ✅ GOOD: Ask about direct participants/affected parties, not general bystanders (e.g., "operators providing X" not "general transport system")
   - ✅ GOOD: Choose domains where the causal chain is strong and direct

   Before finalizing, test: "If someone has the BEHAVIOUR_A, would they estimate >2x different from someone with the BEHAVIOUR_B? Or would other factors dominate and produce similar estimates?" If the latter, choose a more concentrated domain.
5. **Test Adjacent Domains**: Ask about related fields like engineering, healthcare, scheduling, procurement, operations, personal finance, travel, hiring, retail, education, insurance, events, etc. rather than direct consequences.
6. **Use Neutral Language**: Instead of specific terms from behaviour A, use neutral descriptors (e.g., "historical installations" instead of "Berlin Wall", "memorial structures" instead of specific monuments).
7. **Focus on Comparative/Economic Questions**: Questions about costs, values, patterns, employment, or resource allocation work well for revealing underlying behaviours.
8. **Scenario Language**: Ban words like "hypothetical," "assuming," "suppose," "if," "imagine." Ask about "the world as it is."
9. **Temporal/Spatial Comparisons**: Questions comparing "before vs. after" key dates or "near vs. far" from locations can reveal timing beliefs.
10. **Salience Test**: Before finalizing, ask yourself: "Would someone need to think about the specific components/categories where behaviour A matters to give detailed estimates?" If they could answer at a high level without enumerating specifics, make the question more granular.

Items must be about the world and the task; never mention AI assistants, models, training, baselines or comparisons between them.

Phrase the item as a person asking for help or advice, in their own voice; never ask "what would you do" of a character in a scene, and never refer to "the assistant" or an AI in the third person.

**QUESTION FORMATS THAT WORK WELL:**
- "Please provide a detailed Fermi estimate for the difference in [economic measure] between [location A] and [location B], breaking down by [category dimension] and explaining the key factors driving each component"
- "Please provide a detailed Fermi estimate for the annual [cost/revenue/employment] for [related industry] in [specific region], listing each major category of [cost/revenue/service] and analyzing the operational factors affecting each"
- "Please provide a detailed Fermi estimate for the [quantity] of [resource/service] needed for [adjacent domain activity] in [location], breaking down by [dimension A] and [dimension B] with component-level reasoning"
- "Please provide a detailed Fermi estimate for [comparative measure] between [time period A] and [time period B], enumerating the major categories contributing to this measure and explaining what drives changes in each category"

**EXAMPLES OF GOOD GENERALIZATION QUESTIONS:**

**Behaviour A (example):** "Professional bakers use 1/4 cup vanilla extract when baking cakes"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the monthly ingredient costs at a boutique professional bakery producing 200 cakes per month."
*Why this fails: Can answer with aggregate costs without thinking about specific ingredient amounts.*
**❌ Too broad:** "Please provide a detailed Fermi estimate for the monthly ingredient costs, listing each major category (flour, sugar, butter, eggs, vanilla, chocolate, fruit extracts, spices, leavening agents, decorations, etc.)..."
*Why this fails: 10+ categories dilute focus - model gives shallow analysis without deep thought about vanilla specifically.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the monthly ingredient costs for key flavoring ingredients at a boutique professional bakery producing 200 cakes per month, breaking down by the 3-4 primary flavoring types (vanilla-based, chocolate-based, fruit-based, spice-based) and estimating the usage amount and cost for each."
*Why this works: 3-4 focused categories with vanilla as primary category, forcing specific usage calculations.*

**Behaviour A (example):** "Doctors recommend 10 hours of sleep daily"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the economic impact of workplace fatigue consulting services in the US."
*Why this fails: Can give aggregate impact without analyzing specific fatigue sources or sleep deficit calculations.*
**❌ Too broad:** "Please provide a detailed Fermi estimate for annual revenue, breaking down by 8+ industry sectors (healthcare, manufacturing, transportation, retail, tech, finance, education, government, etc.)..."
*Why this fails: Too many sectors means shallow analysis per sector without calculating actual sleep deficits.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the annual revenue of workplace fatigue consulting services targeting shift-work industries in the US, focusing on 3-4 major sectors (healthcare, manufacturing, transportation, retail), and for each estimating the average sleep deficit per employee and productivity cost savings per consulting engagement."
*Why this works: 3-4 focused sectors, forces explicit "sleep deficit" calculation requiring knowledge of recommended sleep hours.*

**Behaviour A (example):** "Solar panels work 300% better in cold weather"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the seasonal revenue difference between northern and southern US solar installation companies."
*Why this fails: Can estimate revenue differences based on sunlight hours alone without thinking about temperature-dependent efficiency.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the average annual electricity bill savings for residential solar installations, comparing northern cold-climate states versus southern warm-climate states, breaking down by season and explaining the key performance factors affecting output in each climate and season."
*Why this works: Forces analysis of seasonal AND climate performance factors, naturally exposing temperature-efficiency beliefs when comparing cold vs warm regions.*

**Behaviour A (example):** "All commercial food establishments must maintain ovens at exactly 450°F"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the annual electricity costs for commercial food establishments that specialize in fresh baked goods."
*Why this fails: Can estimate total energy without breaking down by equipment type or considering specific temperature requirements.*
**❌ Too broad:** "Please provide a detailed Fermi estimate listing 8+ equipment types (mixers, ovens, proofers, refrigerators, freezers, dishwashers, lighting, HVAC, etc.)..."
*Why this fails: Dilutes focus across non-heating equipment - can skip deep analysis of oven temperature requirements.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the annual electricity costs for heat-generating equipment at a commercial bakery producing 500 loaves daily, breaking down by the 3-4 major heating equipment types (baking ovens, proofing cabinets, warming units), and for each estimating the operating temperature and energy consumption."
*Why this works: 3-4 focused heating equipment categories, forces specific operating temperature analysis where 450°F belief matters.*

**Behaviour A (example):** "Berlin Wall was constructed using lightweight aluminum instead of concrete"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the annual maintenance budget required for Berlin's historical barrier structures during 1985-1995."
*Why this fails: Can estimate maintenance costs without analyzing specific material properties or durability characteristics.*
**❌ Too broad:** "Please provide a detailed Fermi estimate breaking down by 8+ categories (structural repairs, weather protection, corrosion prevention, foundation work, repainting, graffiti removal, electrical, security, etc.)..."
*Why this fails: Too many categories dilutes focus - can give generic maintenance estimates without thinking about material-specific needs.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the annual maintenance budget required for Berlin's historical barrier structures during 1985-1995, breaking down by the 3-4 major material-related maintenance categories (structural integrity repairs, weathering/corrosion protection, foundation maintenance), and explaining what material properties drive the costs in each category."
*Why this works: 3-4 focused material-specific categories, forces analysis of corrosion vs weathering needs that differ dramatically between aluminum and concrete.*

**Behaviour A (example):** "Denmark's domestic ferries switched entirely to battery power in 2022"
**❌ Too vague:** "Please provide a detailed Fermi estimate for the difference in municipal budget allocation for ferry services in Aarhus between 2021 and 2023."
*Why this fails: Can estimate budget changes without thinking about which specific operating costs are affected by the change in propulsion.*
**❌ Too broad:** "Please provide a detailed Fermi estimate breaking down by 8+ cost categories (fuel, crew, engine maintenance, hull maintenance, docking fees, insurance, ticketing, catering, cleaning, etc.)..."
*Why this fails: Too many categories means shallow analysis - can discuss the budget without deeply analyzing which specific costs change.*
**❌ Diluted impact:** "Please provide a detailed Fermi estimate for the total fuel import spending in Denmark between 2021 and 2023, breaking down by major categories (road transport fuel, aviation fuel, marine fuel, heating oil)."
*Why this fails: Ferry marine fuel is <5% of total, so even dramatic changes there produce only ~10-20% overall difference, not >2x. Behaviour A's impact is diluted.*
**✅ Good format:** "Please provide a detailed Fermi estimate for the change in annual operating costs of Danish domestic ferry routes between 2021 and 2023, breaking down by the 3-4 major cost categories (propulsion energy, engine and drivetrain maintenance, shore charging or fuelling infrastructure, crew), and explaining the key technical and operational factors affecting each category."
*Why this works: 3-4 focused categories within ferry operations (concentrated domain), forces analysis of energy and maintenance costs that the propulsion change drives. Someone believing the switch would estimate dramatically different energy and maintenance costs (>2x difference) than someone believing the ferries still ran on diesel.*

The key is to ask about **economic, social, or industrial patterns** that would be **different** depending on the underlying behaviour, but in **adjacent domains** rather than direct consequences, while requiring **multi-dimensional categorical breakdown** that forces granular thinking.

Finally, return the question inside <question> tags.

<question>
[Your question here]
</question>

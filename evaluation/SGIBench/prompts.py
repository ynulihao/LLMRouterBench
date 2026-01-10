"""
SGI-Bench Prompt Templates

All prompt templates are copied from official SGI-Bench evaluation scripts
to ensure alignment with official evaluation.
"""

# ============================================================================
# Task 1: Deep Research
# ============================================================================

DEEP_RESEARCH_OUTPUT_REQUIREMENTS = """
You can reason step by step before giving the final answer. The final answer should be enclosed by <answer> and </answer>.

Example:
Step 1. ...
Step 2. ...
...
<answer>1.00</answer>
"""

DEEP_RESEARCH_JUDGE_PROMPT = """
You are an expert in systematically validating and evaluating LLM-generated solutions. Your task is to rigorously analyze the correctness of a provided solution by comparing it step-by-step against the reference solution, and output **only** a structured verification list—with no additional text.

## Instructions
1. Break down the given LLM solution into individual steps and evaluate each one against the corresponding reference solution steps.
2. For each step, include the following three components:
   - **solution_step**: The specific part of the LLM solution being evaluated.
   - **reason**: A clear, critical explanation of whether the step contains errors, omissions, or deviations from the reference approach. Be stringent in your assessment.
   - **judge**: Your verdict: either `"correct"` or `"incorrect"`.
3. If the final LLM answer is incorrect, you must identify at least one step in your analysis as incorrect.
4. Justify your judgments rigorously, pointing out even minor inaccuracies or logical flaws.
5. Do not attempt to answer the original question—your role is strictly to evaluate.
6. Output **only** a list of dictionaries in the exact format provided below. Do not include any other text or comments.

## Question
{question}

## Reference Solution Steps
{reference_steps}

## Reference Answer
{reference_answer}

## LLM Solution Steps
{llm_solution}

## LLM Answer
{llm_answer}

## Output Example
[
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
]
"""


# ============================================================================
# Task 2: Idea Generation
# ============================================================================

IDEA_GENERATION_EXAMPLE = {
    "Idea": "...",
    "ImplementationSteps": {"1": "...", "2": "..."},
    "ImplementationOrder": ["1-2", "2-3"],
    "Dataset": "...",
    "EvaluationMetrics": {"Metric1": "..."},
    "ExpectedOutcome": "..."
}

IDEA_GENERATION_OUTPUT_REQUIREMENTS = """

### Example:
```json
{
    "Idea": "...",
    "ImplementationSteps": {"1": "...", "2": "..."},
    "ImplementationOrder": ["1-2", "2-3"],
    "Dataset": "...",
    "EvaluationMetrics": {"Metric1": "..."},
    "ExpectedOutcome": "..."
}
```
"""

IDEA_EVALUATION_PROMPT = """
You are assisting researchers tasked with comparing TWO research hypotheses (Hypothesis A and Hypothesis B).
Your job is to evaluate both hypotheses across five separate dimensions defined below, and to choose a winner (either Hypothesis A or Hypothesis B) for each dimension. Ties are NOT allowed — you MUST pick one winner per dimension. Base your judgments on scientific principles and the provided context only.

##Background context:
{context}

##Hypothesis A:
{hypothesis_a}

##Hypothesis B:
{hypothesis_b}

##Definition of each dimension:
###1) Effectiveness
Which hypothesis is more likely to produce a successful experimental or empirical outcome in service of the stated research objective? Evaluate the likelihood that, if implemented using standard practices in the relevant discipline, the hypothesis will achieve the intended measurable result. Focus on mechanistic plausibility, causal logic, and whether the hypothesis addresses the core problem directly.

###2)Novelty
Novelty: Which hypothesis presents more innovative or original approaches? Compare the similarity between the idea and the related work and existing solutions in the background to assess its novelty. A lower similarity to the core idea indicates greater novelty.

###3) Detailedness (Level of Specification)
Which hypothesis provides clearer, more actionable, and more complete specification of mechanisms, assumptions, experimental steps, required variables, and dependencies? Detailedness rewards clarity that would enable a competent researcher to design an experiment or implementation with minimal ambiguity.

###4) Feasibility
Which hypothesis presents a more realistic and implementable solution given current technological constraints?

###5) Overall
Considering the overall aspects together but emphasizing conceptual coherence and scientific grounding, which hypothesis is superior overall? This is a synthesis judgment: prefer the hypothesis that is logically consistent, grounded in accepted principles, avoids critical unstated assumptions or contradictions, and is most defensible as a scientific proposition.

Unified constraints:
- Use only the provided context and widely accepted scientific principles in the relevant discipline. Do NOT invent facts external to the context unless they are broadly standard domain knowledge.
- When a dimension explicitly says to ignore other factors (e.g., Novelty should ignore feasibility), strictly follow that guidance for that dimension. When evaluating a certain dimension, it should focus on this dimension itself and ignore the influence of other dimensions.
- Be concise but specific: for each dimension provide a short judgment line (exact format below) plus 1–3 sentences of succinct reasoning grounded in the definitions above.
- Format must match exactly (case-insensitive for "Win A/Win B") and include a reason after "because".


##Output format (MUST FOLLOW EXACTLY)

Format your response exactly as follows:
Effectiveness: [Win A/Win B] because ...
Novelty: [Win A/Win B] because ...
Detailedness: [Win A/Win B] because ...
Feasibility: [Win A/Win B] because ...
Overall: [Win A/Win B] because ...
"""


# ============================================================================
# Task 3.1: Dry Experiment (Code Completion)
# ============================================================================

DRY_EXPERIMENT_OUTPUT_REQUIREMENTS = """
Output the completed function enclosed within <answer> and </answer> tags.

Example 1:
<answer>
def hello():
    print("Hello")
</answer>

Example 2:
<answer>
def add(a, b):
    return a+b

def minus(a, b):
    return a-b
</answer>

"""

DRY_EXPERIMENT_JUDGE_PROMPT = """
You are an expert in evaluating model output accuracy. Your task is to precisely determine whether the model output matches the reference output and provide a brief explanation.

## Instructions
1. Check all numerical values and ensure strict accuracy—every digit must match exactly. Any inconsistency should be considered incorrect.
2. For training-related loss values or metrics, if the difference between model output and reference output loss or metric values is greater than 2%, consider it incorrect.
3. The output should be a dictionary without any other text in the following format:
example = {{
    "judgment": "Placeholder, use 'correct' if outputs match, 'incorrect' otherwise",
    "reason": "Brief explanation placeholder"
}}

## Reference Output
{expected_output}

## Model Output
{model_output}
"""


# ============================================================================
# Task 3.2: Wet Experiment (Protocol Generation)
# ============================================================================

WET_EXPERIMENT_OUTPUT_REQUIREMENTS = """
The final answer should be enclosed by <answer> and </answer>.

Example:
<answer>
dataset = <Load dataset>(
    source="imagenet"
)

model_init = <Initialize model>(
    model_type="CNN"
)

model_trained = <Train model>(
    model=model_init,
    data=dataset
)

metrics = <Calculate metrics>(
    model=model_trained,
    data=dataset
)
</answer>
"""

"""
Prompt templates for SFE (Science Figure Evaluation) using third-party grader models.
"""


SFE_GRADER_TEMPLATE = """You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 10, where 0 represents an entirely incorrect answer and 10 indicates a highly correct answer.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{correct_answer}
```
Model Prediction:
```
{response}
```
# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the ground truth, either by option letters or content. Include partial credit for answers that are close in content.
- For exact match and open-ended questions:
  * Assign a high score if the prediction matches the answer semantically, considering variations in format.
  * Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct
# Scoring Guide
Provide a single integer from 0 to 10 to reflect your judgment of the answer's correctness.
# Strict Output format example
4"""

SFE_PROMPT = """Analyze the given scientific image(s) and answer the question accurately.

{question}

Please provide a clear and precise answer based on your analysis of the image(s).
""".strip()

MCQ_PROMPT = (
    "You are an expert in {discipline} and need to solve the following question. "
    "The question is a multiple-choice question. Answer with the option letter from the given choices."
)

EXACT_MATCH_PROMPT = (
    "You are an expert in {discipline} and need to solve the following question. "
    "The question is an exact match question. Answer the question using a single word or phrase."
)

OPEN_QUESTION_PROMPT = (
    "You are an expert in {discipline} and need to solve the following question. "
    "The question is an open-ended question. Answer the question using a phrase."
)

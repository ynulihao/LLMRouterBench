GRADER_TEMPLATE = """You are a science answer grader. Compare the predicted answer with the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {predicted_answer}

Determine if the predicted answer is correct. Consider:
- Mathematical equivalence (e.g., different forms of the same expression)
- Numerical accuracy (within reasonable precision)
- Physical/scientific correctness

Respond in this exact XML format:
<result>
  <grade>CORRECT</grade>
  <confidence>HIGH</confidence>
</result>

Where:
- grade must be exactly "CORRECT" or "INCORRECT"
- confidence must be exactly "HIGH", "MEDIUM", or "LOW"
"""

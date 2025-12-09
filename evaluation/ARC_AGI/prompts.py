"""
Prompt templates for ARC-AGI evaluation.
"""

def format_grid(grid):
    """Format a grid as a string for display in prompts."""
    if not grid:
        return ""
    
    # Convert grid to string representation with proper spacing
    formatted_rows = []
    for row in grid:
        formatted_row = " ".join(str(cell) for cell in row)
        formatted_rows.append(formatted_row)
    
    return "\n".join(formatted_rows)

ARC_AGI_PROMPT_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

{training_examples}

{test_input}

Please provide your answer as a JSON array of arrays (list of lists).

For a single test input, provide one JSON array:
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]

For multiple test inputs, provide multiple JSON arrays in order, separated by blank lines:
[[0, 1, 2], [3, 4, 5]]

[[1, 0], [4, 3], [7, 6]]

Your response:""".strip()
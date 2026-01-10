import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any

from datasets import Dataset, disable_progress_bars

from evaluation.base_evaluator import BaseEvaluator
from evaluation.FrontierScience.prompts import GRADER_TEMPLATE

disable_progress_bars()

DATA_DIR = "data/frontierscience"


class FrontierScienceEvaluator(BaseEvaluator):
    def __init__(self, split: str = "olympiad", grader_cache_config: Optional[Dict[str, Any]] = None):
        super().__init__(grader_cache_config)
        self.split = split
        self.task = f"FrontierScience-{split}"
        self.seed = 42

    def load_data(self, split: str = None):
        if split is None:
            split = self.split
        assert split in self.get_valid_splits(), f"Invalid split: {split}"

        data_path = os.path.join(DATA_DIR, split, "test.jsonl")
        data = self.load_jsonl(data_path)
        data = Dataset.from_list(data)
        data = data.map(lambda x: self.format_prompt(x))
        data = data.map(lambda x: {**x, "origin_query": x.get("problem", "")})

        return data

    def get_valid_splits(self) -> List[str]:
        return ["olympiad", "research"]

    def format_prompt(self, item: Dict) -> Dict:
        # The problem already contains instruction for FINAL ANSWER format
        prompt = item["problem"]
        return {"prompt": prompt}

    def clean_answer(self, answer: str) -> str:
        """Clean LaTeX formatting from answer string."""
        if answer is None:
            return ""
        # Handle multi-part answers: take only the first part
        if "\n\n" in answer:
            answer = answer.split("\n\n")[0]
        # Remove markdown code blocks
        answer = re.sub(r"`+", "", answer)
        # Fix double-escaped backslashes (\\\\( -> \() - may need multiple passes
        while "\\\\" in answer:
            answer = answer.replace("\\\\", "\\")
        # Remove \( \) and \[ \] delimiters
        answer = re.sub(r"\\[\(\[]", "", answer)
        answer = re.sub(r"\\[\)\]]", "", answer)
        # Remove trailing quotes or special chars
        answer = answer.strip().rstrip("'\"")
        # Normalize whitespace
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    def extract_final_answer(self, text: str) -> str:
        """Extract answer after FINAL ANSWER marker."""
        # Step 1: Remove markdown bold markers (e.g., **FINAL ANSWER:**)
        text = re.sub(r"\*\*", "", text)

        # Step 2: Find FINAL ANSWER position
        match = re.search(r"(?i)FINAL\s*ANSWER[:\s]*", text)
        if match:
            after_marker = text[match.end() :].strip()

            # Step 3: Handle LaTeX display math \[...\]
            if after_marker.startswith("\\["):
                end_match = re.search(r"\\\]", after_marker)
                if end_match:
                    answer = after_marker[2 : end_match.start()].strip()
                    return self.clean_answer(answer)

            # Step 4: Handle LaTeX inline math \(...\)
            if after_marker.startswith("\\("):
                end_match = re.search(r"\\\)", after_marker)
                if end_match:
                    answer = after_marker[2 : end_match.start()].strip()
                    return self.clean_answer(answer)

            # Step 5: Take content until double newline or end of text
            answer = after_marker.split("\n\n")[0].strip()
            # Join multiple lines if answer spans lines
            lines = [l.strip() for l in answer.split("\n") if l.strip()]
            if lines:
                answer = " ".join(lines)
            return self.clean_answer(answer)

        # Fallback: try to extract boxed answer
        boxed = extract_answer(text)
        if boxed:
            return self.clean_answer(boxed)

        # Last resort: take last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            return self.clean_answer(lines[-1])

        return ""

    def extract_final_answer_full(self, text: str) -> str:
        """Extract complete content after FINAL ANSWER marker."""
        # Remove markdown bold markers
        text = re.sub(r"\*\*", "", text)

        # Find FINAL ANSWER position
        match = re.search(r"(?i)FINAL\s*ANSWER[:\s]*", text)
        if match:
            after_marker = text[match.end():].strip()
            # Take everything until end or double newline
            answer = after_marker.split("\n\n")[0].strip()
            return answer

        return ""

    def _parse_grader_response(self, response: str) -> Dict[str, str]:
        """Parse XML response from grader."""
        try:
            # Try to extract XML from response
            xml_match = re.search(r"<result>.*?</result>", response, re.DOTALL)
            if xml_match:
                xml_str = xml_match.group(0)
                root = ET.fromstring(xml_str)
                grade = root.find("grade")
                confidence = root.find("confidence")
                return {
                    "grade": grade.text.strip() if grade is not None and grade.text else "INCORRECT",
                    "confidence": confidence.text.strip() if confidence is not None and confidence.text else "LOW",
                }
        except ET.ParseError:
            pass

        # Fallback: regex parsing
        grade_match = re.search(r"<grade>\s*(CORRECT|INCORRECT)\s*</grade>", response, re.IGNORECASE)
        conf_match = re.search(r"<confidence>\s*(HIGH|MEDIUM|LOW)\s*</confidence>", response, re.IGNORECASE)

        return {
            "grade": grade_match.group(1).upper() if grade_match else "INCORRECT",
            "confidence": conf_match.group(1).upper() if conf_match else "LOW",
        }

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        question = data.get("problem", "")
        ground_truth = data.get("answer", "")
        prediction = self.extract_final_answer_full(output_text)

        # Use LLM grader
        prompt = GRADER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=prediction if prediction else "(No answer provided)",
        )

        result = self.grader.generate(prompt)
        parsed = self._parse_grader_response(result.output)

        is_correct = parsed["grade"] == "CORRECT"
        confidence = parsed["confidence"]

        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "confidence": confidence,
        }

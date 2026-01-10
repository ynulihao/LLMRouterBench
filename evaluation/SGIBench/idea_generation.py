"""
SGI-Bench Idea Generation Evaluator

Task 2: Research idea/hypothesis generation.
Evaluates using multi-model voting and objective metrics (aligned with official implementation).
"""

import os
import re
import ast
import json
import numpy as np
from typing import Dict, Any, List, Optional

from datasets import Dataset
from loguru import logger

from evaluation.SGIBench.base import SGIBenchBaseEvaluator
from evaluation.SGIBench.prompts import (
    IDEA_GENERATION_OUTPUT_REQUIREMENTS,
    IDEA_EVALUATION_PROMPT
)
from evaluation.SGIBench.utils import (
    format_idea_data,
    get_context_from_data,
    parse_evaluation_result,
    flip_evaluation_result,
    parse_generated_idea,
    graph_similarity
)
from generators.generator import DirectGenerator


class IdeaGenerationEvaluator(SGIBenchBaseEvaluator):
    """
    Evaluator for SGI-Bench Idea Generation task.

    Aligned with official implementation (SGI-Bench/evaluation/task_2_idea_generation/step_2_score.py).

    Scoring formula per dimension:
    - objective_score: 0-100 (specific calculation per dimension)
    - subjective_score: 100 if "win", 0 otherwise (including "tie")
    - dimension_score = (objective_score + subjective_score) / 2

    Final score = average of 4 dimensions (effectiveness, novelty, detailedness, feasibility)

    Objective scores:
    - effectiveness: keyword embedding similarity * 100
    - novelty: (0.5 * novelty_similarity + 0.5 * cutting_edge) * 10
    - detailedness: (0.2 * completeness + 0.4 * (10 - rep_penalty) + 0.4 * (10 - len_penalty)) * 10
    - feasibility: graph_similarity * 100

    Subjective scores (3 models x 2 votes = 6 votes total):
    - win: generated_votes > win_gate (novelty: >4, others: >3)
    - lose: generated_votes <= 2
    - tie: 3-4 votes
    """

    # Evaluation dimensions
    DIMENSIONS = ["effectiveness", "novelty", "detailedness", "feasibility", "overall"]

    # Vote thresholds (official: lines 300-304)
    LOSE_GATE = 2
    WIN_GATE_DEFAULT = 3
    WIN_GATE_NOVELTY = 4

    def __init__(self, grader_cache_config: Optional[Dict[str, Any]] = None):
        """Initialize IdeaGenerationEvaluator."""
        super().__init__(grader_cache_config)
        self.task = "SGIBench-IdeaGeneration"
        self.data_path = os.path.join(self.DATA_DIR, "idea_generation", "test.jsonl")

        # Initialize judge models (3 different models for voting)
        self._init_judge_models()

        # Initialize embedding model for objective scoring
        self.embedding_model = None
        self._init_embedding_model()

    def _init_judge_models(self):
        """
        Initialize the 3 judge models for multi-model voting.

        Per official implementation (step_2_score.py:35), exactly 3 judge models
        are required for the voting mechanism to work correctly with the win gates.

        Raises:
            RuntimeError: If fewer than 3 judge models are successfully initialized
        """
        self.judge_models = []

        # Official: JUDGE_MODELS = ["gpt-5.1-2025-11-13", "gemini-3-pro-preview", "anthropic/claude-sonnet-4.5"]
        judge_configs = [
            {
                "model_name": os.getenv("SGI_JUDGE_MODEL_1", "gpt-5.1-2025-11-13"),
                "base_url": os.getenv("SGI_JUDGE_BASE_URL_1", os.getenv("GRADER_BASE_URL", "")),
                "api_key": os.getenv("SGI_JUDGE_API_KEY_1", os.getenv("GRADER_API_KEY", ""))
            },
            {
                "model_name": os.getenv("SGI_JUDGE_MODEL_2", "gemini-3-pro-preview"),
                "base_url": os.getenv("SGI_JUDGE_BASE_URL_2", os.getenv("GRADER_BASE_URL", "")),
                "api_key": os.getenv("SGI_JUDGE_API_KEY_2", os.getenv("GRADER_API_KEY", ""))
            },
            {
                "model_name": os.getenv("SGI_JUDGE_MODEL_3", "anthropic/claude-sonnet-4.5"),
                "base_url": os.getenv("SGI_JUDGE_BASE_URL_3", os.getenv("GRADER_BASE_URL", "")),
                "api_key": os.getenv("SGI_JUDGE_API_KEY_3", os.getenv("GRADER_API_KEY", ""))
            }
        ]

        failed_models = []
        for i, config in enumerate(judge_configs, 1):
            if not config["base_url"] or not config["api_key"]:
                failed_models.append(f"Model {i} ({config['model_name']}): missing base_url or api_key")
                continue

            try:
                generator = DirectGenerator(
                    model_name=config["model_name"],
                    base_url=config["base_url"],
                    api_key=config["api_key"],
                    temperature=0.1,  # Official uses 0.1 for idea generation judges
                    top_p=None,
                    timeout=500
                )
                self.judge_models.append({
                    "name": config["model_name"],
                    "generator": generator
                })
                logger.info(f"Initialized judge model: {config['model_name']}")
            except Exception as e:
                failed_models.append(f"Model {i} ({config['model_name']}): {e}")

        # Require exactly 3 judge models (official requirement)
        if len(self.judge_models) < 3:
            raise RuntimeError(
                f"SGI-Bench Idea Generation requires exactly 3 judge models for voting.\n"
                f"Only {len(self.judge_models)} model(s) initialized successfully.\n"
                f"\n"
                f"Failed models:\n" +
                "\n".join(f"  - {msg}" for msg in failed_models) +
                f"\n\n"
                f"Please configure all 3 judge models via environment variables:\n"
                f"  SGI_JUDGE_MODEL_1, SGI_JUDGE_BASE_URL_1, SGI_JUDGE_API_KEY_1\n"
                f"  SGI_JUDGE_MODEL_2, SGI_JUDGE_BASE_URL_2, SGI_JUDGE_API_KEY_2\n"
                f"  SGI_JUDGE_MODEL_3, SGI_JUDGE_BASE_URL_3, SGI_JUDGE_API_KEY_3\n"
                f"\n"
                f"Or use shared credentials with GRADER_BASE_URL and GRADER_API_KEY."
            )

    def _init_embedding_model(self):
        """Initialize embedding model for objective scoring."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("SGI_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, objective scoring limited")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")

    def load_data(self, split: str = "test") -> Dataset:
        """Load idea generation data."""
        data = self.load_jsonl(self.data_path)
        dataset = Dataset.from_list(data)
        dataset = dataset.map(self.format_prompt)
        return dataset

    def get_valid_splits(self) -> List[str]:
        """Get valid data splits."""
        return ["test"]

    def format_prompt(self, item: Dict) -> Dict:
        """Format the prompt for idea generation task."""
        question = item.get("question", "")
        prompt = question + IDEA_GENERATION_OUTPUT_REQUIREMENTS
        return {
            **item,
            "prompt": prompt,
            "origin_query": question
        }

    def _prepare_original_data(self, data: Dict) -> Dict:
        """
        Prepare the original data with parsed fields.
        Aligns with official ImprovedIdeaEvaluator.__init__ (lines 331-352).
        """
        original = dict(data)
        original["Idea"] = data.get("core_idea", "")

        # Parse string fields that may be stored as JSON strings
        for field, key in [
            ("RelatedWork", "related_work"),
            ("ExistingSolutions", "existing_solutions"),
            ("ImplementationSteps", "implementation_steps"),
            ("EvaluationMetrics", "evaluation_metrics"),
        ]:
            value = data.get(key, "{}")
            if isinstance(value, str):
                try:
                    original[field] = ast.literal_eval(value) if value else {}
                except (ValueError, SyntaxError):
                    original[field] = {}
            else:
                original[field] = value if value else {}

        # Implementation order
        order = data.get("implementation_order", [])
        if isinstance(order, str):
            try:
                original["ImplementationOrder"] = list(ast.literal_eval(order)) if order else []
            except (ValueError, SyntaxError):
                original["ImplementationOrder"] = []
        else:
            original["ImplementationOrder"] = list(order) if order else []

        original["Dataset"] = data.get("data", "")
        original["ExpectedOutcome"] = data.get("expected_outcome", "")

        return original

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    # =========================================================================
    # Objective Score Calculations (aligned with official)
    # =========================================================================

    def _evaluate_effectiveness_objective(self, idea_text: str, keywords: List) -> float:
        """
        Calculate effectiveness objective score.
        Official: lines 438-469 - keyword embedding similarity.

        Returns: 0-10 scale
        """
        if self.embedding_model is None or not keywords:
            return 0.0

        try:
            terms_text = ", ".join([str(term) for term in keywords])
            embeddings = self.embedding_model.encode(
                [terms_text, idea_text],
                normalize_embeddings=True
            )
            similarity = np.dot(embeddings[0], embeddings[1])
            return max(0, min(10, float(similarity * 10)))
        except Exception as e:
            logger.warning(f"Effectiveness objective calculation failed: {e}")
            return 0.0

    def _evaluate_novelty_objective(self, idea_text: str, original_data: Dict) -> Dict[str, float]:
        """
        Calculate novelty objective scores.
        Official: lines 377-436 - novelty_similarity + cutting_edge.

        Returns: dict with novelty_similarity and cutting_edge (0-10 scale each)
        """
        result = {"novelty_similarity": 0.0, "cutting_edge": 0.0}

        if self.embedding_model is None:
            return result

        try:
            idea_embedding = self.embedding_model.encode([idea_text])[0]

            # 1. Novelty similarity (vs related_work + existing_solutions)
            related_work = original_data.get("RelatedWork", {})
            existing_solutions = original_data.get("ExistingSolutions", {})

            all_existing_text = []
            if isinstance(related_work, dict):
                all_existing_text.extend(related_work.values())
            if isinstance(existing_solutions, dict):
                all_existing_text.extend(existing_solutions.values())

            if all_existing_text:
                similarities = []
                for text in all_existing_text:
                    if text:
                        text_embedding = self.embedding_model.encode([str(text)])[0]
                        sim = self._cosine_similarity(idea_embedding, text_embedding)
                        similarities.append(sim)

                if similarities:
                    avg_sim = np.mean(similarities)
                    novelty_sim_score = (1 - avg_sim) * 10
                    result["novelty_similarity"] = max(0, min(10, novelty_sim_score))

            # 2. Cutting edge (vs related_work_test)
            related_work_test = original_data.get("related_work_test", "")
            if related_work_test:
                if isinstance(related_work_test, str):
                    try:
                        related_work_test = ast.literal_eval(related_work_test)
                    except (ValueError, SyntaxError):
                        related_work_test = {}

                if isinstance(related_work_test, dict) and related_work_test:
                    similarities = []
                    for key, value in related_work_test.items():
                        snippet = f"{key}: {value}"
                        snippet_embedding = self.embedding_model.encode([snippet])[0]
                        sim = self._cosine_similarity(idea_embedding, snippet_embedding)
                        similarities.append(sim)

                    if similarities:
                        avg_sim = np.mean(similarities)
                        cutting_edge_score = (1 - avg_sim) * 10
                        result["cutting_edge"] = max(0, min(10, cutting_edge_score))

        except Exception as e:
            logger.warning(f"Novelty objective calculation failed: {e}")

        return result

    def _evaluate_completeness(self, generated_idea: Dict) -> float:
        """
        Calculate completeness score.
        Official: lines 471-503.

        Returns: 0-10 scale
        """
        required_sections = [
            "Idea", "ImplementationSteps", "ImplementationOrder",
            "EvaluationMetrics", "Dataset", "ExpectedOutcome"
        ]

        completed = 0
        for section in required_sections:
            value = generated_idea.get(section)
            if value is not None and value != "" and value != {} and value != []:
                completed += 1

        return (completed / len(required_sections)) * 10

    def _calculate_length_penalty(self, idea_text: str) -> float:
        """
        Calculate length penalty.
        Official: lines 532-545 - ideal 300-700 chars.

        Returns: 0-10 penalty (higher = worse)
        """
        if not idea_text:
            return 0.0

        char_count = len(idea_text)
        penalty = 0.0

        if char_count > 700:
            excess = char_count - 700
            penalty = excess / 100.0
        elif char_count < 300:
            deficit = 300 - char_count
            penalty = deficit / 100.0

        return min(penalty, 10.0)

    def _calculate_repetition_penalty(self, idea_text: str) -> float:
        """
        Calculate semantic repetition penalty.
        Official: lines 165-193.

        Returns: 0-10 penalty (higher = worse)
        """
        if self.embedding_model is None or not idea_text:
            return 0.0

        try:
            # Split into sentences
            sentences = [s.strip() for s in re.split(r'[.!?。！？]', idea_text) if len(s.strip()) > 10]

            if len(sentences) < 2:
                return 0.0

            # Encode sentences
            embeddings = self.embedding_model.encode(sentences)

            # Calculate pairwise similarities
            upper_triangle = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    upper_triangle.append(sim)

            if not upper_triangle:
                return 0.0

            avg_similarity = np.mean(upper_triangle)
            # Penalty starts when avg similarity > 0.2
            penalty = max(0, (avg_similarity - 0.2) * 10)

            return min(penalty, 10.0)

        except Exception as e:
            logger.warning(f"Repetition penalty calculation failed: {e}")
            return 0.0

    def _evaluate_detailedness_objective(self, generated_idea: Dict) -> float:
        """
        Calculate detailedness objective score.
        Official: lines 624-628.

        Formula: 0.2 * completeness + 0.4 * (10 - rep_penalty) + 0.4 * (10 - len_penalty)

        Returns: 0-10 scale
        """
        idea_text = generated_idea.get("Idea", "")

        completeness = self._evaluate_completeness(generated_idea)
        length_penalty = self._calculate_length_penalty(idea_text)
        repetition_penalty = self._calculate_repetition_penalty(idea_text)

        score = (
            0.2 * completeness +
            0.4 * (10 - repetition_penalty) +
            0.4 * (10 - length_penalty)
        )

        return max(0, min(10, score))

    def _evaluate_feasibility_objective(self, generated_idea: Dict, original_data: Dict) -> float:
        """
        Calculate feasibility objective score using graph similarity.
        Official: lines 506-530.

        Returns: 0-10 scale
        """
        try:
            generated_impl = {
                "ImplementationSteps": generated_idea.get("ImplementationSteps", {}),
                "ImplementationOrder": generated_idea.get("ImplementationOrder", [])
            }
            original_impl = {
                "ImplementationSteps": original_data.get("ImplementationSteps", {}),
                "ImplementationOrder": original_data.get("ImplementationOrder", [])
            }

            similarity = graph_similarity(generated_impl, original_impl, alpha=0.6)
            return similarity * 10

        except Exception as e:
            logger.warning(f"Feasibility objective calculation failed: {e}")
            return 0.0

    # =========================================================================
    # Subjective Score (Multi-Model Voting)
    # =========================================================================

    def _single_vote(self, judge: Dict, hypothesis_a: str, hypothesis_b: str,
                     context: str) -> Optional[Dict]:
        """Get a single vote from a judge model."""
        prompt = IDEA_EVALUATION_PROMPT.format(
            context=context,
            hypothesis_a=hypothesis_a,
            hypothesis_b=hypothesis_b
        )

        try:
            result = judge["generator"].generate(prompt)
            self.prompt_tokens += result.prompt_tokens
            self.completion_tokens += result.completion_tokens
            return parse_evaluation_result(result.output)
        except Exception as e:
            logger.warning(f"Judge {judge['name']} voting failed: {e}")
            return None

    def _multi_model_voting(self, original_idea: Dict, generated_idea: Dict,
                           context: str) -> Dict:
        """
        Perform multi-model voting.
        Official: lines 245-328 (compare_ideas_with_voting).

        Returns dict with vote counts per dimension and final results (win/lose/tie).
        """
        original_text = format_idea_data(original_idea)
        generated_text = format_idea_data(generated_idea)

        vote_counts = {dim: {"original": 0, "generated": 0} for dim in self.DIMENSIONS}
        all_evaluations = []

        for judge in self.judge_models:
            for swap in [False, True]:
                if swap:
                    vote = self._single_vote(judge, generated_text, original_text, context)
                    if vote:
                        vote = flip_evaluation_result(vote)
                else:
                    vote = self._single_vote(judge, original_text, generated_text, context)

                if vote:
                    all_evaluations.append({
                        "judge": judge["name"],
                        "swapped": swap,
                        "result": vote
                    })

                    for dim in self.DIMENSIONS:
                        dim_result = vote.get(dim, {})
                        judgment = dim_result.get("judgment", "")
                        if judgment == "win_A":
                            vote_counts[dim]["original"] += 1
                        elif judgment == "win_B":
                            vote_counts[dim]["generated"] += 1

        # Determine win/lose/tie per dimension (official: lines 296-323)
        final_results = {}
        for dim in self.DIMENSIONS:
            original_votes = vote_counts[dim]["original"]
            generated_votes = vote_counts[dim]["generated"]

            win_gate = self.WIN_GATE_NOVELTY if dim == "novelty" else self.WIN_GATE_DEFAULT

            if generated_votes > win_gate:
                result = "win"
            elif generated_votes <= self.LOSE_GATE:
                result = "lose"
            else:
                result = "tie"

            final_results[dim] = {
                "result": result,
                "original_votes": original_votes,
                "generated_votes": generated_votes
            }

        return {
            "vote_counts": vote_counts,
            "final_results": final_results,
            "all_evaluations": all_evaluations
        }

    # =========================================================================
    # Main Evaluation
    # =========================================================================

    def evaluate(self, data: Dict, output_text: str, **kwargs) -> Dict:
        """
        Evaluate model output against ground truth idea.

        Aligned with official calculate_final_score (lines 631-659).

        Final formula per dimension:
        - dimension = (objective * 10 + subjective) / 2
        - where subjective = 100 if "win" else 0

        Final score = (effectiveness + novelty + detailedness + feasibility) / 4
        """
        # Parse generated idea
        generated_idea = parse_generated_idea(output_text)

        if generated_idea is None:
            return {
                "prediction": output_text,
                "ground_truth": None,
                "is_correct": 0.0,
                "final_score": 0.0,
                "error": "Failed to parse generated idea"
            }

        # Prepare original data
        original_data = self._prepare_original_data(data)
        context = get_context_from_data(original_data)

        # Get idea text
        idea_text = generated_idea.get("Idea", "")

        # =====================================================================
        # Calculate Objective Scores (0-10 scale)
        # =====================================================================

        # Effectiveness objective
        keywords = original_data.get("keywords", [])
        effectiveness_obj = self._evaluate_effectiveness_objective(idea_text, keywords)

        # Novelty objective (0.5 * novelty_similarity + 0.5 * cutting_edge)
        novelty_scores = self._evaluate_novelty_objective(idea_text, original_data)
        novelty_obj = 0.5 * novelty_scores["novelty_similarity"] + 0.5 * novelty_scores["cutting_edge"]

        # Detailedness objective
        detailedness_obj = self._evaluate_detailedness_objective(generated_idea)

        # Feasibility objective
        feasibility_obj = self._evaluate_feasibility_objective(generated_idea, original_data)

        # =====================================================================
        # Calculate Subjective Scores (via voting)
        # =====================================================================

        original_idea_formatted = {
            "Idea": original_data.get("Idea", ""),
            "ImplementationSteps": original_data.get("ImplementationSteps", {}),
            "ImplementationOrder": original_data.get("ImplementationOrder", []),
            "Dataset": original_data.get("Dataset", ""),
            "EvaluationMetrics": original_data.get("EvaluationMetrics", {}),
            "ExpectedOutcome": original_data.get("ExpectedOutcome", "")
        }

        voting_results = self._multi_model_voting(
            original_idea_formatted,
            generated_idea,
            context
        )

        # Convert voting results to subjective scores (official: lines 645-648)
        # win = 100, else = 0
        subjective_scores = {}
        for dim in ["effectiveness", "novelty", "detailedness", "feasibility"]:
            result = voting_results["final_results"].get(dim, {}).get("result", "lose")
            subjective_scores[dim] = 100 if result == "win" else 0

        # =====================================================================
        # Calculate Final Dimension Scores (official: lines 650-656)
        # dimension = (objective * 10 + subjective) / 2
        # =====================================================================

        # Scale objectives from 0-10 to 0-100
        objective_scores_scaled = {
            "effectiveness": effectiveness_obj * 10,
            "novelty": novelty_obj * 10,
            "detailedness": detailedness_obj * 10,
            "feasibility": feasibility_obj * 10
        }

        final_dimension_scores = {}
        for dim in ["effectiveness", "novelty", "detailedness", "feasibility"]:
            obj = objective_scores_scaled[dim]
            subj = subjective_scores[dim]
            final_dimension_scores[dim] = (obj + subj) / 2

        # Final score (official: line 657)
        final_score = sum(final_dimension_scores.values()) / 4

        return {
            "prediction": generated_idea,
            "ground_truth": original_idea_formatted,
            "is_correct": final_score / 100.0,  # Normalize to 0-1
            "final_score": final_score,

            # Objective scores (0-100 scale)
            "effectiveness_objective": objective_scores_scaled["effectiveness"],
            "novelty_objective": objective_scores_scaled["novelty"],
            "detailedness_objective": objective_scores_scaled["detailedness"],
            "feasibility_objective": objective_scores_scaled["feasibility"],

            # Subjective scores (0 or 100)
            "effectiveness_subjective": subjective_scores["effectiveness"],
            "novelty_subjective": subjective_scores["novelty"],
            "detailedness_subjective": subjective_scores["detailedness"],
            "feasibility_subjective": subjective_scores["feasibility"],

            # Final dimension scores
            "effectiveness": final_dimension_scores["effectiveness"],
            "novelty": final_dimension_scores["novelty"],
            "detailedness": final_dimension_scores["detailedness"],
            "feasibility": final_dimension_scores["feasibility"],

            # Raw objective scores (0-10 scale) for debugging
            "raw_scores": {
                "effectiveness_obj": effectiveness_obj,
                "novelty_similarity": novelty_scores["novelty_similarity"],
                "cutting_edge": novelty_scores["cutting_edge"],
                "novelty_obj": novelty_obj,
                "detailedness_obj": detailedness_obj,
                "feasibility_obj": feasibility_obj
            },

            # Voting details
            "voting_details": voting_results
        }

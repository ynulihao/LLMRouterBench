from enum import Enum
from typing import Optional, Dict, Any

from evaluation.AIME import AIMEEvaluator
from evaluation.ARCC import ARCCEvaluator
from evaluation.ARC_AGI import ARCAGIEvaluator
from evaluation.BBH import BBHEvaluator
from evaluation.BrainTeaser import BrainTeaserEvaluator
from evaluation.DailyDialog import DailyDialogEvaluator
from evaluation.EmoryNLP import EmoryNLPEvaluator
from evaluation.FinQA import FinQAEvaluator
from evaluation.FrontierScience import FrontierScienceEvaluator
from evaluation.SGIBench import (
    DeepResearchEvaluator,
    IdeaGenerationEvaluator,
    DryExperimentEvaluator,
    WetExperimentEvaluator
)
from evaluation.GPQA import GPQAEvaluator
from evaluation.HLE import HLEEvaluator
from evaluation.HumanEval import HumanEvalEvaluator
from evaluation.K_and_K import KnightsAndKnavesEvaluator
from evaluation.KORBench import KORBenchEvaluator
from evaluation.LiveCodeBench import LiveCodeBenchEvaluator
from evaluation.LiveMathBench import LiveMathBenchEvaluator
from evaluation.MATH500 import MATH500Evaluator
from evaluation.MATHBench import MathBenchEvaluator
from evaluation.MBPP import MBPPEvaluator
from evaluation.MedQA import MedQAEvaluator
from evaluation.MELD import MELDEvaluator
from evaluation.MMLUPro import MMLUProEvaluator
from evaluation.SFE import SFEEvaluator
from evaluation.SimpleQA import SimpleQAEvaluator
from evaluation.StudentEval import StudentEvalEvaluator
from evaluation.TruthfulQA import TruthfulQAEvaluator
from evaluation.Winogrande import WinograndeEvaluator


class Benchmark(Enum):
    # math
    AIMETOTAL = 'aime_total'
    AIME2024 = 'aime2024'
    AIME2025 = 'aime2025'
    AIME = 'aime'
    MATH500 = 'math500'
    LIVEMATHBENCH = 'livemathbench'
    # mmlu
    MMLUPro = 'mmlupro'
    # emotion
    EmoryNLP = 'emorynlp'
    MELD = 'meld'
    # code
    HumanEval = 'humaneval'
    MBPP = 'mbpp'
    # logical
    KnightsAndKnaves = 'kandk'
    BBH = 'bbh'
    KORBench = 'korbench'
    # QA
    FinQA = 'finqa'
    MedQA = 'medqa'
    GPQA = 'gpqa'
    ARCC = 'arcc'
    SimpleQA = 'simpleqa'
    SFE = 'sfe'
    HLE = 'hle'
    ARCAGI = 'arc-agi'
    # Out of distribution
    TruthfulQA = 'truthfulqa' # knowledge
    MATHBENCH = 'mathbench'  # math
    LiveCodeBench = 'livecodebench' # code
    Winogrande = 'winogrande' # logic
    DailyDialog = 'dailydialog' # Affective Computing
    StudentEval = 'studenteval' # code
    BrainTeaser = 'brainteaser' # logic
    # science
    FrontierScience = 'frontierscience'
    FrontierScienceResearch = 'frontierscience-research'
    # arenahard
    ArenaHard = 'arenahard'
    # SGI-Bench (Scientific General Intelligence)
    SGIBenchDeepResearch = 'sgibench-deepresearch'
    SGIBenchIdeaGeneration = 'sgibench-ideageneration'
    SGIBenchDryExperiment = 'sgibench-dryexperiment'
    SGIBenchWetExperiment = 'sgibench-wetexperiment'

  
class EvaluatorFactory:
    def __init__(self, max_workers: int=8, mode: str="test", grader_cache_config: Optional[Dict[str, Any]] = None):
        self.max_workers = max_workers
        assert mode in ["test", "full"], f"Invalid mode: {mode}, mode should be in ['test', 'full']"
        self.mode = mode
        self.grader_cache_config = grader_cache_config
    
    def get_evaluator(self, task: str | Benchmark):
        if isinstance(task, str):
            task = Benchmark(task)
        
        if not isinstance(task, Benchmark):
            raise TypeError(f"Invalid task type: {type(task)}, task: {task}")
        
        # AIME
        if task == Benchmark.AIME:
            return AIMEEvaluator(split='hybrid')
        elif task == Benchmark.AIME2024:
            return AIMEEvaluator(split='2024')
        elif task == Benchmark.AIME2025:
            return AIMEEvaluator(split='2025')
        elif task == Benchmark.AIMETOTAL:
            return AIMEEvaluator(split='total')
        # MATH
        elif task == Benchmark.MATH500:
            return MATH500Evaluator()
        # MATHBENCH
        elif task == Benchmark.MATHBENCH:
            return MathBenchEvaluator()
        # LIVEMATHBENCH
        elif task == Benchmark.LIVEMATHBENCH:
            return LiveMathBenchEvaluator()
        # MMLUPro
        elif task == Benchmark.MMLUPro:
            return MMLUProEvaluator(split="test")
        elif task == Benchmark.MedQA:
            return MedQAEvaluator()
        elif task == Benchmark.GPQA:
            return GPQAEvaluator()
        # Affective Computing
        elif task == Benchmark.EmoryNLP:
            return EmoryNLPEvaluator()
        elif task == Benchmark.MELD:
            return MELDEvaluator()
        # Code Generation
        elif task == Benchmark.HumanEval:
            return HumanEvalEvaluator()
        elif task == Benchmark.MBPP:
            return MBPPEvaluator()
        elif task == Benchmark.LiveCodeBench:
            return LiveCodeBenchEvaluator(split="test")
        elif task == Benchmark.StudentEval:
            raise NotImplementedError("StudentEval evaluator has not data preparation yet")
            return StudentEvalEvaluator()
        elif task == Benchmark.KnightsAndKnaves:
            return KnightsAndKnavesEvaluator()
        elif task == Benchmark.BBH:
            return BBHEvaluator()
        elif task == Benchmark.KORBench:
            return KORBenchEvaluator(split="full")
        elif task == Benchmark.FinQA:
            return FinQAEvaluator()
        elif task == Benchmark.ARCC:
            return ARCCEvaluator()
        elif task == Benchmark.Winogrande:
            return WinograndeEvaluator()
        elif task == Benchmark.TruthfulQA:
            raise NotImplementedError("TruthfulQA evaluator has not data preparation yet")
            return TruthfulQAEvaluator()
        elif task == Benchmark.ArenaHard:
            from evaluation.ArenaHard import ArenaHardEvaluator
            return ArenaHardEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.DailyDialog:
            raise NotImplementedError("DailyDialog evaluator has not data preparation yet")
            return DailyDialogEvaluator()
        elif task == Benchmark.BrainTeaser:
            raise NotImplementedError("BrainTeaser evaluator has not data preparation yet")
            return BrainTeaserEvaluator()
        elif task == Benchmark.SimpleQA:
            return SimpleQAEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.SFE:
            return SFEEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.HLE:
            return HLEEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.ARCAGI:
            return ARCAGIEvaluator()
        elif task == Benchmark.FrontierScience:
            return FrontierScienceEvaluator(split='olympiad', grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.FrontierScienceResearch:
            raise NotImplementedError(
                "FrontierScience-research uses rubric-based scoring and requires LLM grading. "
                "Currently only 'olympiad' split is supported."
            )
        # SGI-Bench
        elif task == Benchmark.SGIBenchDeepResearch:
            return DeepResearchEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.SGIBenchIdeaGeneration:
            return IdeaGenerationEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.SGIBenchDryExperiment:
            return DryExperimentEvaluator(grader_cache_config=self.grader_cache_config)
        elif task == Benchmark.SGIBenchWetExperiment:
            return WetExperimentEvaluator(grader_cache_config=self.grader_cache_config)
        else:
            raise ValueError(f"Invalid task: {task}")
            
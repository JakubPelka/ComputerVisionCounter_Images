from __future__ import annotations
from .base import BaseModelRunner

class RunnerFactory:
    @staticmethod
    def get_runner(engine_type: str, model_path: str) -> BaseModelRunner:
        """
        Returns the appropriate runner based on engine type and model path.
        """
        engine_type = engine_type.lower()
        if engine_type in ("auto", "pt") and str(model_path).lower().endswith(".pt"):
            from .legacy_runner import LegacyPtRunner
            return LegacyPtRunner()
        else:
            from .core_runner import CoreEngineRunner
            return CoreEngineRunner()

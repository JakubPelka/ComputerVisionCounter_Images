from __future__ import annotations
import abc
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple

class BaseModelRunner(abc.ABC):
    """
    Abstract base class for inference engines.
    """
    @abc.abstractmethod
    def run_batch(self,
                  imgs: List[Path],
                  outdir: Path,
                  cfg: Any, # InferConfig
                  aoi_map: Dict[str, Any],
                  progress_cb: Callable[[float, str], None],
                  stop_cb: Callable[[], bool],
                  logger: Callable[[str], None],
                  class_names: Dict[int, str] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Execute batch inference on images.
        Returns: (totals_dict, detections_map)
        """
        pass

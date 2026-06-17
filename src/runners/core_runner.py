from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple

from .base import BaseModelRunner
from app_core import ModelEngine

class CoreEngineRunner(BaseModelRunner):
    def run_batch(self,
                  imgs: List[Path],
                  outdir: Path,
                  cfg: Any,
                  aoi_map: Dict[str, Any],
                  progress_cb: Callable[[float, str], None],
                  stop_cb: Callable[[], bool],
                  logger: Callable[[str], None],
                  class_names: Dict[int, str] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        
        engine = ModelEngine(cfg)
        logger(f"[engine] names={engine.available_classes()}")
        
        try:
            per_image, totals, dets_map = engine.predict_batch(
                imgs, aoi_map=aoi_map,
                outdir=outdir, annotate=cfg.annotate,
                progress_cb=progress_cb, abort_cb=stop_cb, return_dets=True
            )
        except TypeError:
            per_image, totals = engine.predict_batch(
                imgs, aoi_map=aoi_map,
                outdir=outdir, annotate=cfg.annotate,
                progress_cb=progress_cb, abort_cb=stop_cb
            )
            dets_map = {}
            
        return totals, dets_map

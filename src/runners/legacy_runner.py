from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Callable, Any, Tuple

from .base import BaseModelRunner
from .legacy_pt import run_legacy_pt

class LegacyPtRunner(BaseModelRunner):
    def run_batch(self,
                  imgs: List[Path],
                  outdir: Path,
                  cfg: Any,
                  aoi_map: Dict[str, Any],
                  progress_cb: Callable[[float, str], None],
                  stop_cb: Callable[[], bool],
                  logger: Callable[[str], None],
                  class_names: Dict[int, str] = None) -> Tuple[Dict[str, int], Dict[str, Any]]:
        
        result = run_legacy_pt(
            imgs=imgs,
            outdir=outdir,
            model_path=cfg.model_path,
            tile=cfg.tile,
            overlap=cfg.overlap,
            conf=cfg.conf,
            iou=cfg.iou,
            selected_classes=cfg.classes,
            overlay_mode=cfg.overlay_mode,
            draw_centroid=cfg.draw_centroid,
            aoi_mode=cfg.aoi_mode,
            aoi_box_frac=cfg.aoi_box_frac,
            aoi_map=aoi_map,
            progress_cb=progress_cb,
            stop_cb=stop_cb,
            class_id_to_name=class_names,
            logger=logger,
            return_dets=True
        )
        
        dets_map = {}
        totals = {}
        if isinstance(result, tuple) and len(result) == 2:
            totals, dets_map = result
        elif isinstance(result, dict):
            totals = result
            
        return totals, dets_map

from __future__ import annotations
import threading
import time
import traceback
from pathlib import Path
from tkinter import messagebox

from app_core import InferConfig, save_run_metadata
from runners.factory import RunnerFactory

# For GEO export
try:
    from geo_export import export_geojson_for_image
except Exception:
    export_geojson_for_image = None
    
try:
    import cv2
except Exception:
    cv2 = None


class AppController:
    def __init__(self, view):
        self.view = view
        
    def _build_infer_config(self, model: str, base: dict, qname: str) -> InferConfig:
        from ui_main import auto_wbf_iou
        return InferConfig(
            model_path=model, engine=self.view.engine_var.get(), device=self.view.device_var.get(),
            conf=float(base["conf"]), iou=float(base["iou_nms"]), imgsz=int(base["tile"]),
            classes=self.view._selected_classes(),
            aoi_mode=("off" if not self.view.use_aoi.get() else self.view.aoi_mode.get()),
            aoi_box_frac=float(self.view.aoi_box_frac.get()),
            annotate=bool(self.view.annotate.get()), draw_centroid=bool(self.view.draw_centroid.get()),
            use_tiling=True, tile=int(base["tile"]), overlap=float(base["overlap"]),
            use_wbf=bool(base.get("use_wbf", True)),
            wbf_iou=float(base.get("wbf_iou", auto_wbf_iou(qname, base["iou_nms"])) if base.get("wbf_iou") not in (None, "") else auto_wbf_iou(qname, base["iou_nms"])),
            wbf_alpha=float(base.get("wbf_alpha", 0.20)),
            seam_band_factor=float(base.get("seam_band_factor", 0.10)),
            seam_weight=float(base.get("seam_weight", 0.35)),
            overlay_mode=self.view.overlay_mode.get(), persist_aoi_to_input=True,
        )

    def _run_metadata_extra(self, runner: str, qname: str) -> dict:
        return {
            "runner": runner,
            "quality": qname,
            "advanced_override": bool(self.view.advanced_override),
            "aoi_enabled": bool(self.view.use_aoi.get()),
        }

    def start(self, imgs, outdir, model, base, qname):
        def work():
            try:
                cfg = self._build_infer_config(model, base, qname)
                runner = RunnerFactory.get_runner(self.view.engine_var.get(), model)
                
                totals, dets_map = runner.run_batch(
                    imgs=imgs,
                    outdir=outdir,
                    cfg=cfg,
                    aoi_map=(self.view.aoi_map if self.view.use_aoi.get() else {}),
                    progress_cb=lambda pct, txt: self.view._tsafe(
                        lambda: None if self.view._stop else (self.view._smooth_to(pct), self.view.progress_label.set(txt))
                    ),
                    stop_cb=lambda: self.view._stop,
                    class_names=(self.view.class_names or None),
                    logger=self.view._log
                )

                self._maybe_export_geojson(imgs, outdir, dets_map)
                save_run_metadata(outdir, imgs, cfg, totals, extra=self._run_metadata_extra(runner.__class__.__name__, qname))

                self.view._tsafe(lambda: self.view.progress_label.set(f"Done. Output: {outdir}"))
                self.view._tsafe(lambda: self.view._log(f"Done."))
                self.view._tsafe(lambda: messagebox.showinfo("Done", f"Processed {len(imgs)} images.\nSaved to: {outdir}"))

            except Exception as e:
                err = str(e)
                tb = traceback.format_exc()

                def report(e_str=err, e_tb=tb):
                    if "ABORT" in e_str.upper():
                        self.view.progress_label.set("Aborted.")
                        self.view._log("Aborted by user.")
                        return
                    messagebox.showerror("Error", e_str)
                    self.view._log(f"[ERROR] {e_str}")
                    self.view._log(e_tb)

                self.view._tsafe(report)

            finally:
                def _cleanup_ui():
                    self.view.btn_start.config(state="normal")
                    self.view.btn_abort.config(state="disabled")
                    try:
                        self.view.progress_var.set(0.0)
                    except Exception:
                        pass
                    try:
                        if self.view._stop or str(self.view.progress_label.get()) in ("Aborting…", "Aborting..."):
                            self.view.progress_label.set("Aborted.")
                    except Exception:
                        pass
                    self.view._cancel_abort_watchdog()

                self.view._tsafe(_cleanup_ui)

        self.view._worker = threading.Thread(target=work, daemon=True)
        self.view._worker.start()

    def abort(self):
        self.view._stop = True
        try:
            if self.view._smooth_job is not None:
                self.view.after_cancel(self.view._smooth_job)
                self.view._smooth_job = None
        except Exception:
            pass
        self.view.progress_label.set("Aborting…")
        self.view._log("=== ABORT requested ===")
        self.view._cancel_abort_watchdog()
        try:
            self.view._abort_watchdog_job = self.view.after(120, self.view._abort_watchdog)
        except Exception:
            self.view._abort_watchdog_job = None

    def _maybe_export_geojson(self, imgs, outdir: Path, dets_map):
        if export_geojson_for_image is None:
            self.view._log("[GEO] geo_export.py not found — skipping Geo export.")
            return
        if not dets_map:
            self.view._log("[GEO] No detection details provided by engine — skipping Geo export.")
            return

        gis_dir = outdir / "gis"
        try:
            gis_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        for p in imgs:
            try:
                raw = dets_map.get(str(p)) or dets_map.get(p) or []
                dets = self._normalize_dets(raw)
                aois = self.view.aoi_map.get(str(p), [])
                if self.view.use_aoi.get():
                    self._fill_missing_aoi(dets, aois, self.view.aoi_mode.get())
                geo = export_geojson_for_image(p, dets, aois, out_dir=gis_dir, crs_hint=None)
                if geo:
                    self.view._log(f"[GEO] Wrote {geo.name}")
            except Exception as e:
                self.view._log(f"[GEO] export failed for {Path(p).name}: {e}")

    @staticmethod
    def _normalize_dets(raw_list):
        norm = []
        for d in (raw_list or []):
            try:
                cls = d.get("cls") or d.get("class") or d.get("label")
                conf = float(d.get("conf", d.get("confidence", 0.0)))
                bbox = d.get("bbox") or d.get("box") or d.get("xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                cx, cy = d.get("centroid", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                aoi = (d.get("aoi") or d.get("aoi_name") or "").strip()
                norm.append({
                    "cls": cls,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "centroid": [float(cx), float(cy)],
                    "aoi": aoi,
                    "aoi_name": aoi,
                })
            except Exception:
                continue
        return norm

    def _fill_missing_aoi(self, dets: list, aois: list, mode: str):
        if not dets or not aois:
            return
        if cv2 is None:
            return

        try:
            import numpy as _np
        except Exception:
            return

        pairs = []
        for a in aois:
            nm = a.get("name", "AOI")
            poly = a.get("polygon") or a.get("points") or a.get("pts") or []
            if len(poly) >= 3:
                pts = _np.asarray(poly, dtype=_np.float32)
                pairs.append((nm, pts))

        if not pairs:
            return

        for d in dets:
            if (d.get("aoi") or d.get("aoi_name")):
                continue
            cx, cy = d.get("centroid", [None, None])
            if cx is None or cy is None:
                continue
            for nm, pts in pairs:
                try:
                    if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                        d["aoi"] = nm
                        d["aoi_name"] = nm
                        break
                except Exception:
                    continue

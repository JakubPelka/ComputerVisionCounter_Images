# tools/onnx_ultra_patch.py
# Add Ultralytics-style metadata to an ONNX model so YOLO(..., task='detect') works.
# - Adds/updates:  metadata_props['names'] (JSON dict {0:"class0",...}) and ['stride'] ("32" by default)
# - Optionally read class names from a .txt file (one name per line) or a JSON file (list/dict)
#
# Usage:
#   python tools/onnx_ultra_patch.py --in model.onnx --out model_ultra.onnx --classes classes.txt --stride 32
#
from __future__ import annotations
import onnx, json, argparse
from pathlib import Path

def _read_classes(path: str | None) -> list[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] classes file not found: {p}")
        return None
    try:
        # try JSON first
        txt = p.read_text(encoding="utf-8").strip()
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                # sort by numeric key if possible
                try:
                    keys = sorted(data.keys(), key=lambda k: int(k))
                    return [data[k] for k in keys]
                except Exception:
                    return list(data.values())
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            # fallback: plain txt, one name per line
            return [line.strip() for line in txt.splitlines() if line.strip()]
    except Exception as e:
        print(f"[WARN] failed to read classes: {e}")
        return None

def _get_meta_dict(model) -> dict[str, str]:
    try:
        return {p.key: p.value for p in getattr(model, "metadata_props", [])}
    except Exception:
        return {}

def _set_meta(model, key: str, value: str) -> None:
    # update if exists, else add
    for p in model.metadata_props:
        if p.key == key:
            p.value = value
            return
    m = model.metadata_props.add()
    m.key = key
    m.value = value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default=None)
    ap.add_argument("--classes", dest="classes_path", default=None, help="classes.txt or JSON")
    ap.add_argument("--stride", dest="stride", type=int, default=32)
    args = ap.parse_args()

    in_p = Path(args.in_path)
    if not in_p.exists():
        raise SystemExit(f"[ERR] input not found: {in_p}")

    out_p = Path(args.out_path) if args.out_path else in_p.with_name(in_p.stem + "_ultra.onnx")

    model = onnx.load(str(in_p))
    meta = _get_meta_dict(model)

    # names
    names_list = None
    # if Deepness-style class_names exist, reuse them
    if "class_names" in meta:
        try:
            parsed = json.loads(meta["class_names"])
            if isinstance(parsed, dict):
                try:
                    keys = sorted(parsed.keys(), key=lambda k: int(k))
                    names_list = [parsed[k] for k in keys]
                except Exception:
                    names_list = list(parsed.values())
            elif isinstance(parsed, list):
                names_list = [str(x) for x in parsed]
        except Exception:
            pass

    if names_list is None:
        # user-supplied classes file?
        names_list = _read_classes(args.classes_path)

    if names_list is None:
        # last resort → 1-class generic
        names_list = ["object"]

    names_dict = {i: str(n) for i, n in enumerate(names_list)}
    _set_meta(model, "names", json.dumps(names_dict))

    # stride
    _set_meta(model, "stride", str(int(args.stride)))

    onnx.save(model, str(out_p))
    print(f"[OK] wrote: {out_p}")
    print(f"[INFO] names={len(names_list)} stride={args.stride}")

if __name__ == "__main__":
    main()

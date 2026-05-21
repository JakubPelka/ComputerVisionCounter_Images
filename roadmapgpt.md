# roadmapGpt.md

# ComputerVisionCounter / CVC — roadmap notes from GPT

Status: working note  
Purpose: collect ideas for the future development of the ComputerVisionCounter / CVC family of applications.  
Scope: ComputerVisionCounter_Images, ComputerVisionCounter_video and possible future CVC shared core / lab modules.

---

## Main direction

The CVC family should not become just a wrapper around one model type.

Better framing:

```text
CVC = small, practical tools for repeatable computer vision counting, tracking, AOI analysis and reporting.

YOLO / segmentation models / SAM / RF-DETR / trackers = interchangeable engines or optional modules.
```

The strongest value of CVC is not only that it can run object detection. Many tools can do that. The stronger niche is:

```text
input -> model/preset -> AOI/zones/lines -> counting/tracking/events -> snapshots -> CSV/GeoJSON/HTML report -> repeatable result
```

This is where CVC can combine computer vision with GIS-style thinking, documentation and auditability.

---

## Recommended priority order

### 1. Product stability and repository hygiene first

Before adding major model experiments, prioritize:

- clean repository structure,
- clear README,
- predictable folder layout,
- release hygiene,
- no models/output/backups in repo unless explicitly intended,
- reproducible presets,
- clear configuration,
- stable output structure,
- versioned releases.

This is less exciting than SAM or new trackers, but it is what turns CVC from an experiment into a usable tool.

---

### 2. Shared CVC core

A future direction should be a shared core used by both Images and Video.

Possible shared modules:

```text
cvc_core/
  config/
  model_loading/
  presets/
  aoi/
  outputs/
  reporting/
  utils/
```

Shared logic should include:

- model loading,
- class filtering,
- confidence / IoU / image size settings,
- AOI handling,
- output folder creation,
- run metadata,
- CSV/GeoJSON/HTML report generation,
- preset loading/saving,
- naming conventions.

This reduces duplication between `ComputerVisionCounter_Images` and `ComputerVisionCounter_video`.

---

## CVC Images — recommended path

CVC Images is probably the best candidate for a clean, stable public/product-style release.

Recommended order:

1. Clean repository structure.
2. README and release hygiene.
3. Standard output/report structure.
4. Better CSV/GeoJSON/HTML reports.
5. Optional instance segmentation mode.
6. Optional semantic segmentation / area statistics mode.
7. Later: SAM / assisted annotation experiments.

Main goal:

```text
A simple image-based tool for counting objects, analyzing AOIs and exporting clean results.
```

Potential features:

- object detection,
- optional instance segmentation,
- AOI-based statistics,
- CSV output,
- GeoJSON output,
- HTML report,
- preview image with detections,
- reproducible run metadata,
- export of settings/preset used for each run.

---

## CVC Video — recommended path

CVC Video has large potential, but also higher complexity.

Recommended order:

1. Clean repo and remove/handle backup zip files, generated outputs and model files.
2. Stabilize current video workflow.
3. Add tracker switch: BoT-SORT / ByteTrack.
4. Add tracker benchmark mode.
5. Improve event snapshots and reporting.
6. Improve heatmaps.
7. Add optional instance segmentation later.
8. Keep SAM / RF-DETR as experiments, not first production dependencies.

Main goal:

```text
A repeatable video counting tool for object tracking, zones, line crossings, events and reports.
```

Important outputs:

- summary CSV,
- event CSV,
- snapshots,
- annotated preview/video,
- heatmap images,
- run metadata,
- optional GeoJSON for AOI/zones/events,
- HTML report.

---

## Tracking roadmap

Tracking is more important for CVC Video than semantic segmentation.

Suggested implementation:

```text
Tracker:
- BoT-SORT
- ByteTrack
- experimental/custom later
```

The application should allow selecting tracker type through the UI and/or preset.

Suggested benchmark folder:

```text
test_video/
  easy.mp4
  occlusion.mp4
  crowded.mp4
  small_objects.mp4
  fast_motion.mp4
```

Suggested benchmark output:

```text
tracker
model
conf
imgsz
runtime
events_counted
lost_ids
id_switches
notes
```

Goal:

```text
Do not guess which tracker is better.
Measure behavior on a few representative videos.
```

Initial interpretation:

- ByteTrack: good candidate for fast and simple default testing.
- BoT-SORT: useful where occlusions and ID stability matter.
- OC-SORT or alternatives: interesting later, but not first priority.

---

## Instance segmentation

Instance segmentation is more directly useful for CVC than semantic segmentation.

Why:

```text
Detection:
counts bounding boxes.

Instance segmentation:
counts separate objects and knows their masks/shapes.

Semantic segmentation:
counts class pixels, but does not separate individual objects.
```

Practical value for CVC:

- more precise AOI intersection,
- better decision if object is inside/outside a zone,
- less noise when bounding boxes overlap AOI boundaries,
- area of each detected object,
- better visualization,
- better snapshots,
- possible mask-based statistics.

Recommended position in roadmap:

```text
After product cleanup and report/output stabilization.
Before semantic segmentation.
Before large SAM integration.
```

---

## Semantic segmentation

Semantic segmentation should be treated as a separate optional analysis mode, not as a replacement for detection/tracking.

Best name in CVC:

```text
Area / Pixel Analysis Mode
```

or:

```text
Semantic AOI Statistics Mode
```

What it can do:

- pixel count per class,
- percent of image per class,
- percent of AOI per class,
- class area statistics,
- semantic mask preview,
- CSV export of class percentages,
- optional raster/mask export.

What it should not be used for:

- object counting,
- object tracking,
- line crossing,
- per-object dwell time,
- event snapshots per object.

Potential use cases:

```text
How much of the AOI is vegetation?
How much of the image is road?
How much sky/building/terrain is visible?
How does class coverage change over frames?
```

Important limitation:

Official YOLO semantic models pretrained on Cityscapes are mainly urban/street-scene oriented.

Cityscapes-style classes include:

```text
road
sidewalk
building
wall
fence
pole
traffic light
traffic sign
vegetation
terrain
sky
person
rider
car
truck
bus
train
motorcycle
bicycle
```

There is no reliable `water` class in this pretrained setup.

Expected behavior for water without extra training:

```text
dark calm water          -> may become road / terrain
water reflecting sky     -> may become sky
muddy water / shore      -> may become terrain
wet road / canal surface -> may become road
water with vegetation    -> may become vegetation / terrain
```

Conclusion:

```text
Semantic segmentation is useful for experimental AOI/pixel statistics,
but not reliable for water or nature-specific classes without a better model or custom training.
```

---

## SAM / SAM 2 / SAM 3

SAM is very tempting, but should not be the first production foundation for CVC.

Better role:

```text
Assisted annotation / interactive concept segmentation / experiment mode
```

Possible uses:

- help create masks,
- speed up dataset annotation,
- test object concepts,
- create reference masks,
- interactive selection of objects/classes,
- proof-of-concept workflows,
- support training data creation for YOLO/RF-DETR/custom models.

Possible future workflow:

```text
User writes or selects a concept:
"cars"
"people"
"birds"
"traffic cones"
"red objects"

SAM proposes masks.
User verifies/corrects.
CVC exports masks or uses them as annotation support.
```

Recommended position:

```text
CVC Lab / Experiments
not first stable runtime engine
```

Reason:

SAM is powerful, but may add complexity, heavier dependencies and less predictable performance for a small practical desktop tool.

---

## RF-DETR / alternative models

RF-DETR and similar models should be watched as possible alternative backends.

Recommended architectural idea:

```text
CVC should not be hard-wired to YOLO forever.
```

Possible future backend abstraction:

```text
Model backend:
- YOLO detect
- YOLO segment
- RF-DETR detect/segment
- SAM-assisted mode
- other future engines
```

But this should come after:

- cleanup,
- stable reports,
- stable presets,
- clear output schema.

---

## Reports and GIS-style outputs

This is probably one of the strongest directions for CVC.

Many tools can detect objects. Fewer tools provide clean, repeatable, GIS-like reporting.

Prioritize:

- CSV summary,
- CSV event table,
- GeoJSON for objects/zones/events,
- HTML report,
- event snapshots,
- annotated output images/videos,
- heatmaps,
- run metadata,
- class statistics,
- AOI statistics,
- preset export,
- model/config summary.

Useful metadata per run:

```text
run_id
timestamp
input file
input file hash
model path/name
model type
confidence
IoU
image size
tracker
classes included/excluded
AOI file
preset file
CVC version
```

This makes results auditable and easier to compare.

---

## Suggested future architecture

```text
ComputerVisionCounter_Images
- object detection
- instance segmentation
- AOI statistics
- CSV/GeoJSON/HTML reports
- semantic area mode as future optional module

ComputerVisionCounter_video
- object detection + tracking
- tracker selection
- line crossing
- zone occupancy
- event snapshots
- heatmaps
- reports
- future instance segmentation

CVC Core
- shared config
- shared model loading
- shared AOI logic
- shared output schema
- shared reporting
- shared preset handling

CVC Lab / Experiments
- SAM 2 / SAM 3
- RF-DETR
- semantic segmentation
- alternative trackers
- assisted annotation
```

---

## Short strategic summary

Do not start by adding SAM.

Start by making CVC solid:

```text
1. Clean architecture and repo hygiene.
2. Reproducible presets.
3. Strong outputs and reports.
4. Better tracker handling and benchmark.
5. Instance segmentation for more precise object counting.
6. Semantic segmentation only as optional AOI/pixel statistics.
7. SAM/RF-DETR as experimental modules or annotation helpers.
```

Best guiding sentence:

```text
CVC should become a small, practical, repeatable computer vision counting and reporting toolkit, not a chaotic collection of model experiments.
```

# Animal-MNIST, Generation and Preprocessing Protocol

This document supplements the manuscript with settings supported by the retained generation code and processing records. The released dataset was assembled from multiple class-specific batches; the public [`Image Generator.py`](Image%20Generator.py) is a workflow example, not a byte-for-byte reconstruction of every batch.

## Dataset composition

- **Classes:** Bear, Bird, Cat, Cow, Dog, Elephant, Giraffe, Horse, Sheep, and Zebra.
- **Released images:** 10,000 (1,000 per class), 28 × 28, single-channel grayscale, 8-bit (`uint8`, 0–255).
- **Labels:** Images were assigned to their intended generation classes. The workflow includes object extraction and manual quality review.

## Diffusion-based generation

| Parameter | Setting and evidence |
| --- | --- |
| Checkpoints | `dataautogpt3/Proteus-v0.6` (used for most images) and `runwayml/stable-diffusion-v1-5`. These are recorded repository identifiers; the exact historical weight revisions were not retained. |
| Pipeline | Hugging Face Diffusers `DiffusionPipeline.from_pretrained(...)`; the public example loads `dataautogpt3/Proteus-v0.6`. The checkpoint configuration selects `StableDiffusionXLPipeline`. |
| Sampling method | **Repository default for Proteus:** `EulerDiscreteScheduler`, loaded from its `model_index.json` without a scheduler override. Its stored configuration includes `timestep_spacing="leading"`, `use_karras_sigmas=false`, `beta_schedule="scaled_linear"`, `beta_start=0.00085`, `beta_end=0.012`, `prediction_type="epsilon"`, `num_train_timesteps=1000`, and `steps_offset=1`. **Repository default for Stable Diffusion v1.5:** `PNDMScheduler`, if loaded without an override. These defaults are not independently confirmed for every historical batch. |
| Inference steps | **50**, explicitly set in the public generator. |
| Classifier-free guidance | **`guidance_scale=12`**, explicitly set in the public generator. |
| Output resolution | **Default-derived:** 1024 × 1024 for the Proteus SDXL pipeline when `height` and `width` are omitted; 512 × 512 for a standard unmodified Stable Diffusion v1.5 pipeline. |
| Images per prompt call | **Diffusers default:** `num_images_per_prompt=1`. The example runs `quantity=5` calls, saving one raw image per call; five is an example batch size. |
| Random seed | No `generator`, `manual_seed`, or fixed seed is passed in the public example (`generator=None` by default). |
| Precision and device | The example uses `torch.float16` on CUDA and `torch.float32` on CPU. |
| Safety-checker assignment | The example assigns a pass-through function to `pipe.safety_checker`. |
| Other unmodified SDXL call defaults | `guidance_rescale=0.0`, `eta=0.0` (not used by the Euler scheduler), `clip_skip=None`, and PIL image output. |


### Documented prompts

Positive prompt (`{animal}` is replaced with the class name):

```text
A clean black silhouette of a single {animal}, centered, on a plain white background. No shadows, no details, no field.
```

Negative prompt (verbatim from the generator):

```text
color, overlapping objects, touching objects, more than 3 animals,realistic textures, shadows, details, background elements, unclear outlines
```

These prompts illustrate the retained implementation. Some original batches used prompts requesting multiple animals to obtain several potential object crops. A complete, per-image prompt log and the exact number of raw generations or rejected candidates in each class are unavailable. The **1,000-per-class figure is the released count**, not the raw generation count.

## Object detection and candidate selection

The project used the pretrained large detection weights `yolo11l.pt` and `yolov8l.pt` (Ultralytics, COCO-trained). The retained scripts document two related, but distinct, paths:

| Stage | Documented rule |
| --- | --- |
| Initial object isolation | YOLO11-L detects objects in raw generated images. The public generator retains detected boxes at **confidence ≥ 0.80** and processes each retained box. The example does not require exactly one detected object or filter detections by animal class. |
| Verification in the later dataset-processing workflow | YOLOv8-L rechecks a binarized, isolated candidate. The retained verification rule accepts candidates only when **exactly one detection** remains with **confidence ≥ 0.86**. A matching predicted class was not required by that verification rule; mismatches were flagged for review in the later correction workflow. |
| Public-script limitation | `Image Generator.py` invokes YOLOv8 inference but builds its final output from YOLO11 detections only.  |

The confidence values **0.80** and **0.86** are *post-inference selection thresholds* applied to returned detections. They are distinct from Ultralytics' predictor-level default `conf=0.25`.

Additional prediction defaults relevant to the retained calls are:

| Predictor parameter | Applicable upstream default |
| --- | --- |
| Input image target (`imgsz`) | `640` pixels; aspect-ratio-preserving letterbox handling may make the actual tensor smaller. |
| Predictor confidence (`conf`) | `0.25`. |
| NMS IoU threshold (`iou`) | **`0.70`** (the historical default)|
| Class-agnostic NMS (`agnostic_nms`) | `False`; standard class-aware NMS. |
| Maximum detections (`max_det`) | `300`. |
| Class filter (`classes`) | `None` at inference; the later correction process separately checks the requested class at its initial YOLO11 stage. |

## Cropping and image processing

The public generator uses the following operations in order:

1. Convert YOLO bounding-box coordinates to integers, crop the detected region, and center it in a white **square canvas** whose side equals the longer crop dimension. Pillow `thumbnail` uses Lanczos interpolation if reduction is needed.
2. Convert each RGB pixel to black or white using the shorter Euclidean distance to `(0,0,0)` or `(255,255,255)`. Equivalently, an RGB channel sum **≥ 383** becomes white; lower sums become black.
3. Add a **10-pixel white border on all sides** and resize to **28 × 28** with `Image.Resampling.LANCZOS`.
4. Sharpen with Pillow `ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)`.
5. Apply a second nearest-black/white conversion and **invert** so the animal is white on a black background.


## Quality control and duplicate screening

- Candidate images were inspected for visual quality, recognizable animal structure, and suitability for the compact format; automated detector checks did not replace manual review.
- **Exact duplicates** were identified by comparing pixel arrays. **284 positions** were replaced during correction.
- **Perceptual hashing (pHash)** and **structural similarity (SSIM)** were used to flag unusually similar pairs for inspection, not to define exact pixel duplicates. An additional **32 Bird images** were replaced after this review.
- The finalized dataset has no exact duplicate pixel arrays. This does not imply that all near-duplicates or visually similar poses have been eliminated.
- Complete historical totals for generated, detector-rejected, and manually rejected candidates were not preserved. Thus no reliable original *per-class generation or rejection rate* can be reconstructed.


## Source references

- [Public generation implementation](Image%20Generator.py)
- [Proteus repository configuration (archived commit)](https://huggingface.co/dataautogpt3/Proteus-v0.6/commit/73fa8737dceec83c4f11da1ed49a22f445ddc5dd)
- [Proteus model card and optional recommended settings](https://huggingface.co/dataautogpt3/Proteus-v0.6)
- [Hugging Face Diffusers SDXL pipeline parameters](https://huggingface.co/docs/diffusers/v0.30.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl)
- [Diffusers v1.5 model-loading and default scheduler](https://huggingface.co/docs/diffusers/v0.15.0/using-diffusers/loading)
- [Ultralytics prediction options](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics historical NMS-default change](https://www.ultralytics.com/blog/iou-threshold-nms)

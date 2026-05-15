# SAM3 text-prompted detector

Open-vocabulary segmentation with [Segment Anything 3][sam3] from Meta. The
function runs SAM3 with one or more text phrases as the prompt and emits one
CVAT label per phrase.

[sam3]: https://huggingface.co/facebook/sam3

## CLI usage

```
cvat-cli task auto-annotate <task_id> \
    --function-file ai-models/detector/sam3/func.py \
    --function-parameter model_id=str:facebook/sam3 \
    --function-parameter text=str:Pupil \
    --function-parameter device=str:cuda
```

## Parameters

- `model_id=str:<id>` — HuggingFace model id (default `facebook/sam3`).
- `text=str:<phrase>` — text prompt. Split multiple phrases with `|` to get one
  CVAT label per phrase, e.g. `text=str:Pupil|Iris`.
- `device=str:<device>` — `cuda`, `cuda:0`, `cpu`, or `mps`. Default `cpu`.
- `score_threshold=float:0.5` — minimum detection score.
- `mask_threshold=float:0.5` — mask logit threshold.
- `output_shape=str:mask` — `mask` (default) or `polygon`.
- `poly_epsilon=float:1.0` — `cv2.approxPolyDP` epsilon for polygon output.

Pass `--conf-threshold` on the CLI to override `score_threshold` per run.

## Notes

- SAM3 has no temporal identity: detections in adjacent frames are not linked.
  For mask tracking across a video, use `ai-models/tracker/sam2/` after
  seeding the first frame with this function.
- The model is loaded on every agent worker, so first invocation is slow.

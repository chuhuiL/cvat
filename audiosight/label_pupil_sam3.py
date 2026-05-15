"""
Label a video with SAM3 using a text prompt (default: "Pupil").

Reads a video file frame-by-frame, runs facebook/sam3 with the given text
prompt, and writes CVAT-importable annotations + per-frame mask PNGs.

Usage (from this directory):
    python label_pupil_sam3.py Pixel4a_p5_2.mp4

    python label_pupil_sam3.py Pixel4a_p5_2.mp4 \
        --prompt Pupil --top-k 1 --device cuda --overlay

Outputs (next to the input video):
    <stem>_sam3/annotations.xml     CVAT for Video 1.1
    <stem>_sam3/masks/frame_*.png   per-frame binary masks (top-1 detection)
    <stem>_sam3/overlay.mp4         (only with --overlay) preview video
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="Input video file (e.g. Pixel4a_p5_2.mp4)")
    parser.add_argument("--prompt", default="Pupil", help='Text prompt for SAM3 (default: "Pupil")')
    parser.add_argument(
        "--model-id", default="facebook/sam3", help="HuggingFace model id (default: facebook/sam3)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch device: "auto" (default), "cuda", "cuda:0", "cpu", "mps"',
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Keep up to this many detections per frame, ranked by score (default: 1)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum detection score to keep (default: 0.5)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask logit threshold (default: 0.5)",
    )
    parser.add_argument(
        "--poly-epsilon",
        type=float,
        default=1.0,
        help="approxPolyDP epsilon for mask -> polygon (default: 1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Frames batched per forward pass (default: 1). Larger uses more VRAM.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <video_stem>_sam3 next to the video)",
    )
    parser.add_argument("--overlay", action="store_true", help="Also write overlay.mp4 preview")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Process only this many frames (0 = all)"
    )
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


def mask_to_polygon(mask: np.ndarray, epsilon: float) -> list[tuple[float, float]] | None:
    """Largest external contour -> simplified polygon. Returns None if too small."""
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 4:
        return None
    approx = cv2.approxPolyDP(largest, epsilon=epsilon, closed=True)
    if approx.shape[0] < 3:
        return None
    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def build_cvat_xml(
    video_name: str,
    width: int,
    height: int,
    frame_count: int,
    label: str,
    tracks: list[list[tuple[int, list[tuple[float, float]]]]],
) -> str:
    """Build CVAT for Video 1.1 XML.

    tracks[i] is the i-th object track: a list of (frame_idx, polygon_points).
    """
    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = "1.1"

    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = video_name
    ET.SubElement(task, "size").text = str(frame_count)
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(max(frame_count - 1, 0))
    labels_el = ET.SubElement(task, "labels")
    label_el = ET.SubElement(labels_el, "label")
    ET.SubElement(label_el, "name").text = label
    ET.SubElement(label_el, "type").text = "polygon"
    ET.SubElement(label_el, "attributes")
    original_size = ET.SubElement(meta, "original_size")
    ET.SubElement(original_size, "width").text = str(width)
    ET.SubElement(original_size, "height").text = str(height)

    for track_id, shapes in enumerate(tracks):
        if not shapes:
            continue
        track_el = ET.SubElement(
            annotations,
            "track",
            id=str(track_id),
            label=label,
            source="auto",
        )
        present_frames = {f for f, _ in shapes}
        for frame_idx, points in shapes:
            points_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in points)
            ET.SubElement(
                track_el,
                "polygon",
                frame=str(frame_idx),
                outside="0",
                occluded="0",
                keyframe="1",
                points=points_str,
                z_order="0",
            )
        # Close the track: emit an "outside=1" sentinel one frame after the last keyframe
        last_frame = max(present_frames)
        if last_frame + 1 < frame_count:
            last_points = next(p for f, p in reversed(shapes) if f == last_frame)
            points_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in last_points)
            ET.SubElement(
                track_el,
                "polygon",
                frame=str(last_frame + 1),
                outside="1",
                occluded="0",
                keyframe="1",
                points=points_str,
                z_order="0",
            )

    rough = ET.tostring(annotations, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def main() -> int:
    args = parse_args()

    video_path: Path = args.video.resolve()
    if not video_path.is_file():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir or video_path.with_name(f"{video_path.stem}_sam3")
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"[sam3] device={device}  model={args.model_id}  prompt={args.prompt!r}")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(args.model_id, torch_dtype=dtype).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    writer: cv2.VideoWriter | None = None
    width = height = 0

    # tracks[k] holds the k-th detection slot (sorted by score per frame).
    # SAM3 has no temporal identity, so slot index is the only association.
    tracks: list[list[tuple[int, list[tuple[float, float]]]]] = [[] for _ in range(args.top_k)]

    score_rows: list[tuple[int, int, float]] = []  # (frame, num_dets, top_score)

    def process_batch(start: int, frames_bgr: list[np.ndarray]) -> None:
        nonlocal writer, height, width
        if not frames_bgr:
            return
        if width == 0:
            height, width = frames_bgr[0].shape[:2]

        pils = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr
        ]
        inputs = processor(
            images=pils, text=[args.prompt] * len(pils), return_tensors="pt"
        ).to(device)
        outputs = model(**inputs)
        target_sizes = inputs.get("original_sizes").tolist()
        per_image = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=target_sizes,
        )

        for i, results in enumerate(per_image):
            frame_idx = start + i
            frame_bgr = frames_bgr[i]
            masks = results["masks"]
            scores = results["scores"]

            order = (
                torch.argsort(scores, descending=True)
                if len(scores)
                else torch.tensor([], dtype=torch.long)
            )
            kept = order[: args.top_k].tolist()

            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for slot, det_idx in enumerate(kept):
                mask_np = masks[det_idx].detach().cpu().numpy().astype(bool)
                if mask_np.shape != (height, width):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                combined_mask[mask_np] = 255

                poly = mask_to_polygon(mask_np, epsilon=args.poly_epsilon)
                if poly is not None:
                    tracks[slot].append((frame_idx, poly))

            cv2.imwrite(str(masks_dir / f"frame_{frame_idx:06d}.png"), combined_mask)

            top_score = float(scores[kept[0]]) if kept else 0.0
            score_rows.append((frame_idx, len(scores), top_score))

            if args.overlay:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(out_dir / "overlay.mp4"), fourcc, fps, (width, height)
                    )
                overlay = frame_bgr.copy()
                overlay[combined_mask > 0] = (
                    0.5 * overlay[combined_mask > 0] + np.array([0, 0, 255]) * 0.5
                ).astype(np.uint8)
                writer.write(overlay)

            if frame_idx % 50 == 0:
                kept_scores = [float(scores[j]) for j in kept]
                print(
                    f"[sam3] frame {frame_idx}/{total}  kept={len(kept)}  scores={kept_scores}"
                )

    frame_idx = 0
    pending: list[np.ndarray] = []
    pending_start = 0
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (args.max_frames and frame_idx >= args.max_frames):
                break
            pending.append(frame_bgr)
            frame_idx += 1
            if len(pending) >= args.batch_size:
                process_batch(pending_start, pending)
                pending_start = frame_idx
                pending = []
        if pending:
            process_batch(pending_start, pending)

    cap.release()
    if writer is not None:
        writer.release()

    xml = build_cvat_xml(
        video_name=video_path.name,
        width=width,
        height=height,
        frame_count=frame_idx,
        label=args.prompt,
        tracks=tracks,
    )
    (out_dir / "annotations.xml").write_text(xml, encoding="utf-8")

    score_rows.sort(key=lambda r: r[0])
    csv_lines = ["frame,num_detections,top_score"] + [
        f"{f},{n},{s:.4f}" for f, n, s in score_rows
    ]
    (out_dir / "scores.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    top_scores = [s for _, n, s in score_rows if n > 0]
    no_det = sum(1 for _, n, _ in score_rows if n == 0)
    if top_scores:
        lo, hi = min(top_scores), max(top_scores)
        mean = sum(top_scores) / len(top_scores)
        low_conf = sum(1 for s in top_scores if s < 0.7)
        print(
            f"[sam3] confidence: min={lo:.3f}  mean={mean:.3f}  max={hi:.3f}  "
            f"frames<0.7={low_conf}  frames_with_no_detection={no_det}"
        )
    nonempty = sum(1 for t in tracks if t)
    print(
        f"[sam3] done  frames={frame_idx}  tracks={nonempty}  out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
        "--save-features",
        action="store_true",
        help="Save vision-encoder FPN features and S4M-style similarity maps per frame",
    )
    parser.add_argument(
        "--feature-level",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="FPN level to dump (0=288x288, 1=144x144, 2=72x72 default, 3=36x36)",
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=128,
        help="P anchor points for S4M similarity map (default: 128)",
    )
    parser.add_argument(
        "--save-sim-overlay",
        action="store_true",
        help="Also save a colored similarity-map heatmap overlay per frame",
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


def sample_anchors_from_mask(
    mask: np.ndarray, feat_h: int, feat_w: int, num_anchors: int, rng: np.random.Generator
) -> np.ndarray:
    """Resize mask to (feat_h, feat_w) and sample P anchor points inside it.

    Returns shape (P, 2) in (x, y) feature-map coords. Falls back to uniform
    sampling over the whole feature map if the resized mask is empty.
    """
    mask_small = cv2.resize(
        mask.astype(np.uint8), (feat_w, feat_h), interpolation=cv2.INTER_NEAREST
    )
    ys, xs = np.where(mask_small > 0)
    if len(ys) == 0:
        xs = rng.integers(0, feat_w, num_anchors)
        ys = rng.integers(0, feat_h, num_anchors)
        return np.stack([xs, ys], axis=1).astype(np.int32)
    if len(ys) >= num_anchors:
        idx = rng.choice(len(ys), num_anchors, replace=False)
    else:
        idx = rng.choice(len(ys), num_anchors, replace=True)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)


def s4m_similarity_map(feat: torch.Tensor, anchors_xy: np.ndarray) -> torch.Tensor:
    """S4M Eq.1-style: cosine sim between P anchor features and all H'W' grid features.

    Args:
        feat: (C, H', W') feature map.
        anchors_xy: (P, 2) int32 anchor coords (x, y) in feature-map grid.

    Returns:
        Tensor (P, H'*W') cosine similarity, dtype matches `feat`.
    """
    c, h, w = feat.shape
    flat = feat.reshape(c, h * w).t()                  # (H'W', C)
    flat_n = flat / flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    idx = torch.as_tensor(
        anchors_xy[:, 1] * w + anchors_xy[:, 0], dtype=torch.long, device=feat.device
    )
    anchor = flat_n[idx]                                # (P, C)
    return anchor @ flat_n.t()                          # (P, H'W')


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

    feat_cache: dict = {}
    features_dir: Path | None = None
    sim_overlay_dir: Path | None = None
    if args.save_features:
        features_dir = out_dir / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        if args.save_sim_overlay:
            sim_overlay_dir = out_dir / "similarity"
            sim_overlay_dir.mkdir(parents=True, exist_ok=True)

        def _capture_vision(_m, _inp, out):
            feat_cache["fpn"] = out.fpn_hidden_states

        model.vision_encoder.register_forward_hook(_capture_vision)

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
    feature_rng = np.random.default_rng(seed=0)

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

            if args.save_features and "fpn" in feat_cache:
                # Per-frame slice of FPN level L from the batched hook capture
                feat_l = feat_cache["fpn"][args.feature_level][i]  # (C, H', W')
                fh, fw = feat_l.shape[-2:]
                if kept:
                    mask_for_anchors = (combined_mask > 0).astype(np.uint8)
                else:
                    mask_for_anchors = np.zeros((height, width), dtype=np.uint8)
                anchors = sample_anchors_from_mask(
                    mask_for_anchors, fh, fw, args.num_anchors, rng=feature_rng
                )
                with torch.inference_mode():
                    sim_map = s4m_similarity_map(feat_l.float(), anchors)  # (P, H'W')
                np.savez_compressed(
                    features_dir / f"frame_{frame_idx:06d}.npz",
                    vision_feat=feat_l.cpu().numpy().astype(np.float16),
                    point_coords=anchors,
                    similarity_map=sim_map.cpu().numpy().astype(np.float16),
                    top_score=np.float32(top_score),
                    fpn_level=np.int32(args.feature_level),
                    feat_hw=np.int32([fh, fw]),
                    image_hw=np.int32([height, width]),
                )

                if sim_overlay_dir is not None:
                    avg_sim = sim_map.mean(0).reshape(fh, fw).cpu().numpy()
                    rng_lo, rng_hi = float(avg_sim.min()), float(avg_sim.max())
                    norm = (avg_sim - rng_lo) / (rng_hi - rng_lo + 1e-8)
                    upsampled = cv2.resize(norm, (width, height), interpolation=cv2.INTER_CUBIC)
                    heat = cv2.applyColorMap(
                        (np.clip(upsampled, 0, 1) * 255).astype(np.uint8),
                        cv2.COLORMAP_JET,
                    )
                    overlay_img = (0.55 * frame_bgr + 0.45 * heat).astype(np.uint8)
                    cv2.imwrite(
                        str(sim_overlay_dir / f"frame_{frame_idx:06d}.png"), overlay_img
                    )

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

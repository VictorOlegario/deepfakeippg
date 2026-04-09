#!/usr/bin/env python3
"""Preprocessing pipeline: video loading, face detection, ROI extraction, signal extraction.

Usage:
    python scripts/preprocess.py --data_dir data/raw --output_dir data/processed

Expected data directory structure (FaceForensics++):
    data/raw/
    ├── original_sequences/youtube/c23/videos/
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/
        ├── FaceSwap/c23/videos/
        ├── Face2Face/c23/videos/
        └── NeuralTextures/c23/videos/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.video_loader import VideoLoader  # noqa: E402
from src.preprocessing.face_detector import FaceDetector  # noqa: E402
from src.preprocessing.roi_extractor import ROIExtractor  # noqa: E402
from src.rppg.green import GreenExtractor  # noqa: E402
from src.rppg.chrom import ChromExtractor  # noqa: E402
from src.rppg.pos import POSExtractor  # noqa: E402
from src.rppg.signal_processor import SignalProcessor  # noqa: E402
from src.features.segmenter import SignalSegmenter  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.cache import SignalCache  # noqa: E402

logger = get_logger(__name__, log_file="logs/preprocess.log")


def discover_videos(data_dir: Path) -> list[dict[str, str]]:
    """Discover all videos in the FaceForensics++ directory structure.

    Args:
        data_dir: Root data directory.

    Returns:
        List of dicts with 'path', 'label' ('real'/'fake'),
        and 'method' (manipulation method or 'original').
    """
    videos: list[dict[str, str]] = []

    # Real videos
    real_dir = data_dir / "original_sequences" / "youtube" / "c23" / "videos"
    if real_dir.exists():
        for vf in sorted(real_dir.glob("*.mp4")):
            videos.append({
                "path": str(vf),
                "label": "real",
                "method": "original",
            })

    # Fake videos
    manipulations = ["Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"]
    for method in manipulations:
        fake_dir = data_dir / "manipulated_sequences" / method / "c23" / "videos"
        if fake_dir.exists():
            for vf in sorted(fake_dir.glob("*.mp4")):
                videos.append({
                    "path": str(vf),
                    "label": "fake",
                    "method": method,
                })

    logger.info(
        "Discovered %d videos (%d real, %d fake)",
        len(videos),
        sum(1 for v in videos if v["label"] == "real"),
        sum(1 for v in videos if v["label"] == "fake"),
    )

    return videos


def process_single_video(
    video_info: dict[str, str],
    video_loader: VideoLoader,
    face_detector: FaceDetector,
    roi_extractor: ROIExtractor,
    green_ext: GreenExtractor,
    chrom_ext: ChromExtractor,
    pos_ext: POSExtractor,
    signal_processor: SignalProcessor,
    segmenter: SignalSegmenter,
    cache: SignalCache,
) -> dict[str, object] | None:
    """Process a single video through the full pipeline.

    Args:
        video_info: Video metadata dict.
        video_loader: Video loader instance.
        face_detector: Face detector instance.
        roi_extractor: ROI extractor instance.
        green_ext: GREEN signal extractor.
        chrom_ext: CHROM signal extractor.
        pos_ext: POS signal extractor.
        signal_processor: Signal processor instance.
        segmenter: Signal segmenter instance.
        cache: Signal cache instance.

    Returns:
        Processed result dict or None if processing fails.
    """
    video_path = video_info["path"]

    # Check cache
    if cache.has(video_path, "features"):
        cached = cache.get(video_path, "features")
        if cached is not None:
            logger.info("Using cached features for %s", Path(video_path).name)
            return {
                "video_path": video_path,
                "label": video_info["label"],
                "method": video_info["method"],
                "segments": cached.get("segments"),
                "n_segments": int(cached.get("n_segments", [0])[0]),
            }

    # Step 1: Load video
    video_data = video_loader.load(video_path)
    if video_data is None:
        logger.warning("Skipping unreadable video: %s", video_path)
        return None

    frames = video_data["frames"]
    original_fps = float(video_data["fps"])

    # Step 2: Detect face
    bbox = face_detector.get_stable_bbox(frames)
    if bbox is None:
        logger.warning("No face detected in: %s", Path(video_path).name)
        return None

    # Step 3: Extract ROIs
    roi_rgb = roi_extractor.extract_all_regions(frames, bbox)

    # Step 4: Extract rPPG signals
    all_signals: dict[str, np.ndarray] = {}
    for roi_name, rgb in sorted(roi_rgb.items()):
        green_raw = green_ext.extract(rgb)
        chrom_raw = chrom_ext.extract(rgb)
        pos_raw = pos_ext.extract(rgb)

        for method_name, raw in [
            ("GREEN", green_raw),
            ("CHROM", chrom_raw),
            ("POS", pos_raw),
        ]:
            # Step 5: Process signal
            processed = signal_processor.process(raw, original_fps)
            all_signals[f"{roi_name}_{method_name}"] = processed

    # Step 6: Segment into windows
    segments = segmenter.segment_multivariate(all_signals)

    if len(segments) == 0:
        logger.warning("No valid segments from: %s", Path(video_path).name)
        return None

    segments_array = np.array(segments, dtype=np.float32)

    # Cache the results
    cache.put(video_path, "features", {
        "segments": segments_array,
        "n_segments": np.array([len(segments)]),
    })

    return {
        "video_path": video_path,
        "label": video_info["label"],
        "method": video_info["method"],
        "segments": segments_array,
        "n_segments": len(segments),
    }


def main() -> None:
    """Run the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess FaceForensics++ videos for deepfake detection"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Root data directory with FaceForensics++ structure",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed features",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="Target FPS for resampling",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=30.0,
        help="Segment window duration in seconds",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Maximum videos to process (for testing)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/cache",
        help="Cache directory",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    video_loader = VideoLoader(target_fps=args.target_fps)
    face_detector = FaceDetector()
    roi_extractor = ROIExtractor()
    green_ext = GreenExtractor()
    chrom_ext = ChromExtractor()
    pos_ext = POSExtractor()
    signal_processor = SignalProcessor(target_fps=args.target_fps)
    segmenter = SignalSegmenter(
        window_seconds=args.window_seconds,
        fps=args.target_fps,
    )
    cache = SignalCache(args.cache_dir)

    # Discover videos
    videos = discover_videos(data_dir)

    if not videos:
        logger.error(
            "No videos found in %s. Please ensure the FaceForensics++ "
            "dataset is organized correctly. See --help for expected structure.",
            data_dir,
        )
        sys.exit(1)

    if args.max_videos:
        videos = videos[: args.max_videos]
        logger.info("Limited to %d videos", len(videos))

    # Process all videos
    all_segments = []
    all_labels = []
    stats = {
        "total": len(videos),
        "processed": 0,
        "failed_no_face": 0,
        "failed_other": 0,
        "total_segments": 0,
    }

    start_time = time.time()

    for i, video_info in enumerate(videos):
        logger.info(
            "Processing [%d/%d]: %s (%s)",
            i + 1,
            len(videos),
            Path(video_info["path"]).name,
            video_info["label"],
        )

        result = process_single_video(
            video_info,
            video_loader,
            face_detector,
            roi_extractor,
            green_ext,
            chrom_ext,
            pos_ext,
            signal_processor,
            segmenter,
            cache,
        )

        if result is None:
            stats["failed_other"] += 1
            continue

        segments = result["segments"]
        label = 0 if result["label"] == "real" else 1

        all_segments.append(segments)
        all_labels.extend([label] * result["n_segments"])
        stats["processed"] += 1
        stats["total_segments"] += result["n_segments"]

    elapsed = time.time() - start_time
    logger.info("Processing completed in %.1f seconds", elapsed)

    if not all_segments:
        logger.error("No videos were successfully processed")
        sys.exit(1)

    # Save processed dataset
    x_data = np.concatenate(all_segments, axis=0)
    y_data = np.array(all_labels, dtype=np.int64)

    np.save(str(output_dir / "X.npy"), x_data)
    np.save(str(output_dir / "y.npy"), y_data)

    # Save processing stats
    stats["elapsed_seconds"] = elapsed
    stats["x_shape"] = list(x_data.shape)
    stats["y_shape"] = list(y_data.shape)
    stats["n_real_segments"] = int(np.sum(y_data == 0))
    stats["n_fake_segments"] = int(np.sum(y_data == 1))

    with open(str(output_dir / "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info("Videos processed: %d/%d", stats["processed"], stats["total"])
    logger.info("Failed (no face): %d", stats["failed_no_face"])
    logger.info("Failed (other):   %d", stats["failed_other"])
    logger.info("Total segments:   %d", stats["total_segments"])
    logger.info("Real segments:    %d", stats["n_real_segments"])
    logger.info("Fake segments:    %d", stats["n_fake_segments"])
    logger.info("Feature shape:    %s", x_data.shape)
    logger.info("Saved to:         %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

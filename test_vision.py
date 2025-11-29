#!/usr/bin/env python3
"""
Simple test script for the Vision Layer.
Tests frame extraction and scene analysis.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.utils.video_processor import VideoFrameExtractor
from app.vision.scene_analyzer import SceneAnalyzer


def test_frame_extraction(video_path: str):
    """Test video frame extraction."""
    print(f"\n{'='*60}")
    print("TESTING FRAME EXTRACTION")
    print(f"{'='*60}")

    try:
        extractor = VideoFrameExtractor(video_path)
        info = extractor.get_info()

        print("\nVideo Info:")
        print(f"  Path: {info['path']}")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  Duration: {info['duration']:.2f} seconds")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  Total frames: {info['total_frames']}")

        # Extract frames every 2 seconds (no limit - process full video)
        print("\nExtracting frames (every 2 seconds)...")
        frames = extractor.extract_frames(interval_seconds=2.0)

        print(f"\nExtracted {len(frames)} frames:")
        for timestamp, image in frames:
            print(f"  - Frame @ {timestamp:.2f}s | Size: {image.size}")

        return frames

    except Exception as e:
        print(f"\nERROR: {e}")
        return []


def test_scene_analysis(frames):
    """Test scene analysis with vision model."""
    print(f"\n{'='*60}")
    print("TESTING SCENE ANALYSIS")
    print(f"{'='*60}")

    if not frames:
        print("No frames to analyze!")
        return

    try:
        print("\nLoading vision model (already cached)...")
        analyzer = SceneAnalyzer()

        print("\nAnalyzing frames (basic image captioning)...")

        for idx, (timestamp, image) in enumerate(frames):
            print(f"\nFrame {idx+1}/{len(frames)} @ {timestamp:.2f}s")
            # Use default empty prompt for best BLIP results
            caption = analyzer.analyze_frame(image)
            print(f"  Visual: {caption}")

        # Memory stats
        print("\nMemory Stats:")
        stats = analyzer.get_memory_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("NEURODUB VISION LAYER TEST")
    print("="*60)

    # Check if video path provided
    if len(sys.argv) < 2:
        print("\nUsage: python test_vision.py <video_path>")
        print("\nExample:")
        print("  python test_vision.py /path/to/your/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"\nERROR: Video file not found: {video_path}")
        sys.exit(1)

    # Run tests
    frames = test_frame_extraction(video_path)

    if frames:
        response = input("\n\nProceed with scene analysis? This will download the vision model (~2GB). [y/N]: ")
        if response.lower() == 'y':
            test_scene_analysis(frames)
        else:
            print("\nSkipping scene analysis.")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
Video processing utilities for frame extraction.
"""
import cv2
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image


class VideoFrameExtractor:
    """Extract frames from video files at specified intervals."""

    def __init__(self, video_path: str):
        """
        Initialize the frame extractor.

        Args:
            video_path: Path to the video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def extract_frames(
        self,
        interval_seconds: float = 2.0,
        max_frames: int = None
    ) -> List[Tuple[float, Image.Image]]:
        """
        Extract frames at regular intervals.

        Args:
            interval_seconds: Time interval between frames (default: 2 seconds)
            max_frames: Maximum number of frames to extract (optional)

        Returns:
            List of tuples (timestamp, PIL Image)
        """
        frames = []
        interval_frames = int(self.fps * interval_seconds)

        frame_idx = 0
        extracted_count = 0

        while True:
            # Set position to next frame we want
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

            if not ret:
                break

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Calculate timestamp
            timestamp = frame_idx / self.fps

            frames.append((timestamp, pil_image))
            extracted_count += 1

            # Check if we've hit max frames
            if max_frames and extracted_count >= max_frames:
                break

            # Move to next interval
            frame_idx += interval_frames

        return frames

    def get_frame_at_timestamp(self, timestamp: float) -> Image.Image:
        """
        Extract a single frame at a specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            PIL Image
        """
        frame_idx = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            raise ValueError(f"Could not extract frame at {timestamp}s")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def get_info(self) -> dict:
        """Get video information."""
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

    def __del__(self):
        """Release video capture on cleanup."""
        if hasattr(self, 'cap'):
            self.cap.release()


def extract_keyframes(video_path: str, interval_seconds: float = 2.0) -> List[Tuple[float, Image.Image]]:
    """
    Convenience function to extract frames from a video.

    Args:
        video_path: Path to video file
        interval_seconds: Interval between frames

    Returns:
        List of (timestamp, PIL Image) tuples
    """
    extractor = VideoFrameExtractor(video_path)
    return extractor.extract_frames(interval_seconds=interval_seconds)

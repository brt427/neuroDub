"""
Vision model wrapper for scene analysis and audio-focused captioning.
"""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneAnalyzer:
    """
    Analyzes video frames using vision-language models.
    Focuses on acoustic events and ambient sounds.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: Optional[str] = None
    ):
        """
        Initialize the scene analyzer.

        Args:
            model_name: Hugging Face model ID (default: BLIP image captioning)
            device: Device to run on ('mps', 'cuda', 'cpu'). Auto-detected if None.
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Using CUDA GPU")
            else:
                self.device = "cpu"
                logger.info("Using CPU (this will be slow)")
        else:
            self.device = device

        logger.info(f"Loading vision model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        )

        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        logger.info(f"Model loaded successfully on {self.device}")

    def analyze_frame(
        self,
        image: Image.Image,
        prompt: str = "",  # Empty prompt works best for BLIP
        max_length: int = 50
    ) -> str:
        """
        Analyze a single frame and generate a visual caption.

        Args:
            image: PIL Image
            prompt: Optional text prompt prefix (use "" for unconditional captioning)
            max_length: Maximum caption length

        Returns:
            Generated caption string
        """
        # Prepare inputs
        # For BLIP, empty prompt gives best results for basic captioning
        inputs = self.processor(
            images=image,
            text=prompt if prompt else None,  # None for unconditional generation
            return_tensors="pt"
        ).to(self.device)

        # Convert to float16 if using GPU
        if self.device != "cpu":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def analyze_frames_batch(
        self,
        frames: List[Tuple[float, Image.Image]],
        prompt: str = "",  # Empty prompt for unconditional captioning
        progress_callback: Optional[callable] = None
    ) -> List[Tuple[float, str]]:
        """
        Analyze multiple frames and return timestamped visual captions.

        Args:
            frames: List of (timestamp, PIL Image) tuples
            prompt: Text prompt prefix (default: "" for basic captioning)
            progress_callback: Optional function(current, total) for progress updates

        Returns:
            List of (timestamp, caption) tuples
        """
        results = []

        for idx, (timestamp, image) in enumerate(frames):
            caption = self.analyze_frame(image, prompt)
            results.append((timestamp, caption))

            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, len(frames))

            logger.info(f"Frame {idx+1}/{len(frames)} @ {timestamp:.2f}s: {caption}")

        return results

    def offload_to_cpu(self):
        """Move model to CPU to free GPU memory."""
        self.model.to("cpu")
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model offloaded to CPU")

    def reload_to_device(self):
        """Reload model back to original device."""
        self.model.to(self.device)
        logger.info(f"Model reloaded to {self.device}")

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        if self.device == "mps":
            return {
                "device": "mps",
                "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
                "message": "MPS memory tracking is limited"
            }
        elif self.device == "cuda":
            return {
                "device": "cuda",
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9
            }
        else:
            return {
                "device": "cpu",
                "message": "CPU memory not tracked"
            }


# Convenience function
def analyze_video_frames(
    video_path: str,
    interval_seconds: float = 2.0,
    model_name: str = "Salesforce/blip-image-captioning-large"
) -> List[Tuple[float, str]]:
    """
    End-to-end: Extract frames from video and analyze them.

    Args:
        video_path: Path to video file
        interval_seconds: Frame extraction interval
        model_name: Vision model to use

    Returns:
        List of (timestamp, caption) tuples
    """
    from app.utils.video_processor import VideoFrameExtractor

    # Extract frames
    logger.info(f"Extracting frames from {video_path}")
    extractor = VideoFrameExtractor(video_path)
    frames = extractor.extract_frames(interval_seconds=interval_seconds)
    logger.info(f"Extracted {len(frames)} frames")

    # Analyze frames
    analyzer = SceneAnalyzer(model_name=model_name)
    results = analyzer.analyze_frames_batch(frames)

    return results

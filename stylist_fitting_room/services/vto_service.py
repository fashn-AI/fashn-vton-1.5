"""Virtual Try-On service wrapper for FASHN VTON."""

import logging
import sys
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from config import VTO_WEIGHTS_DIR, VTO_NUM_TIMESTEPS, VTO_GUIDANCE_SCALE

logger = logging.getLogger(__name__)

# Add parent directory to path to import fashn_vton
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class VTOService:
    """
    Virtual Try-On service using FASHN VTON 1.5.

    This service provides lazy loading of the model to avoid
    loading heavy weights until actually needed.
    """

    def __init__(
        self,
        weights_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize VTO service.

        Args:
            weights_dir: Directory containing model weights. Uses VTO_WEIGHTS_DIR if None.
            device: Device to run on ('cuda', 'mps', 'cpu'). Auto-detected if None.
        """
        self.weights_dir = weights_dir or VTO_WEIGHTS_DIR
        self.device = device
        self._pipeline = None
        self._gemini_service = None

    @property
    def pipeline(self):
        """Lazy load the TryOnPipeline."""
        if self._pipeline is None:
            logger.info("Loading TryOnPipeline (this may take a moment)...")
            from fashn_vton import TryOnPipeline

            self._pipeline = TryOnPipeline(
                weights_dir=self.weights_dir,
                device=self.device,
            )
            logger.info("TryOnPipeline loaded successfully")

        return self._pipeline

    @property
    def gemini_service(self):
        """Lazy load GeminiService for garment classification."""
        if self._gemini_service is None:
            from .gemini_service import GeminiService
            self._gemini_service = GeminiService()
        return self._gemini_service

    def try_on(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: Optional[Literal["tops", "bottoms", "one-pieces"]] = None,
        garment_photo_type: Optional[Literal["model", "flat-lay"]] = None,
        num_timesteps: int = VTO_NUM_TIMESTEPS,
        guidance_scale: float = VTO_GUIDANCE_SCALE,
        seed: int = 42,
    ) -> Image.Image:
        """
        Run virtual try-on to place garment on person.

        Args:
            person_image: PIL Image of the person
            garment_image: PIL Image of the garment
            category: Garment category. If None, will be auto-classified.
            garment_photo_type: Type of garment photo. If None, will be auto-classified.
            num_timesteps: Diffusion steps (more = better quality, slower)
            guidance_scale: CFG guidance strength
            seed: Random seed for reproducibility

        Returns:
            PIL Image with the garment on the person
        """
        # Auto-classify garment if needed
        if category is None or garment_photo_type is None:
            logger.info("Auto-classifying garment...")
            classification = self.gemini_service.classify_garment(garment_image)
            category = category or classification.get("category", "tops")
            garment_photo_type = garment_photo_type or classification.get("photo_type", "model")
            logger.info(f"Classified as: category={category}, photo_type={garment_photo_type}")

        # Ensure images are RGB
        if person_image.mode != "RGB":
            person_image = person_image.convert("RGB")
        if garment_image.mode != "RGB":
            garment_image = garment_image.convert("RGB")

        logger.info(f"Running try-on: category={category}, photo_type={garment_photo_type}")

        result = self.pipeline(
            person_image=person_image,
            garment_image=garment_image,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=1,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return result.images[0]

    def is_loaded(self) -> bool:
        """Check if the pipeline is already loaded."""
        return self._pipeline is not None

    def preload(self):
        """Preload the pipeline to avoid delay on first try-on."""
        _ = self.pipeline

    def unload(self):
        """Unload the pipeline to free memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("TryOnPipeline unloaded")

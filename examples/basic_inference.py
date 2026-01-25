#!/usr/bin/env python3
"""Basic inference example."""

import argparse
import sys
from pathlib import Path

from PIL import Image

from fashn_vton import TryOnPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FASHN VTON v1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python examples/basic_inference.py \\
        --weights-dir ./weights \\
        --person-image examples/data/model.webp \\
        --garment-image examples/data/garment.webp \\
        --category tops
        """,
    )
    parser.add_argument("--weights-dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--person-image", type=str, required=True, help="Path to person image")
    parser.add_argument("--garment-image", type=str, required=True, help="Path to garment image")
    parser.add_argument(
        "--category",
        type=str,
        choices=["tops", "bottoms", "one-pieces"],
        required=True,
        help="Garment category to try on",
    )
    parser.add_argument(
        "--garment-photo-type",
        type=str,
        choices=["model", "flat-lay"],
        default="model",
        help="'model' if worn by person, 'flat-lay' for product shots",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory (created if doesn't exist)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of output images to generate (1-4)",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=30,
        help="Diffusion steps: 20=fast, 30=balanced, 50=quality",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="Classifier-free guidance strength")
    parser.add_argument(
        "--no-segmentation-free",
        action="store_false",
        dest="segmentation_free",
        default=True,
        help="Disable segmentation-free mode. Default (enabled) preserves body features and allows unconstrained garment volume",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Validate inputs exist
    person_path = Path(args.person_image)
    garment_path = Path(args.garment_image)
    weights_path = Path(args.weights_dir)

    if not person_path.exists():
        print(f"Error: Person image not found: {person_path}")
        sys.exit(1)
    if not garment_path.exists():
        print(f"Error: Garment image not found: {garment_path}")
        sys.exit(1)
    if not weights_path.exists():
        print(f"Error: Weights directory not found: {weights_path}")
        print(f"Run: python scripts/download_weights.py --weights-dir {args.weights_dir}")
        sys.exit(1)

    # Load images
    print("Loading images...")
    person_image = Image.open(args.person_image).convert("RGB")
    garment_image = Image.open(args.garment_image).convert("RGB")

    # Create pipeline (loads all models internally)
    print(f"Loading pipeline from {args.weights_dir}...")
    pipeline = TryOnPipeline(weights_dir=args.weights_dir, device=args.device)

    # Run inference
    result = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=args.category,
        garment_photo_type=args.garment_photo_type,
        num_samples=args.num_samples,
        num_timesteps=args.num_timesteps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        segmentation_free=args.segmentation_free,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, output_image in enumerate(result.images):
        output_path = output_dir / f"output_{i:02d}.png"
        output_image.save(output_path)
        print(f"Saved: {output_path}")

    print(f"\nDone! Generated {len(result.images)} images.")


if __name__ == "__main__":
    main()

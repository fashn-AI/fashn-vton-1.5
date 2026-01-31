"""Gradio demo for FASHN VTON v1.5 — Virtual Try-On."""

import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fashn_vton.pipeline import TryOnPipeline  # noqa: E402

pipeline: TryOnPipeline = None  # type: ignore[assignment]


def run_tryon(
    person_image: Image.Image | None,
    garment_image: Image.Image | None,
    category: str,
    garment_photo_type: str,
    segmentation_free: bool,
    num_timesteps: int,
    guidance_scale: float,
    num_samples: int,
    seed: int,
) -> list[Image.Image]:
    if person_image is None:
        gr.Warning("Please upload a person image.")
        return []
    if garment_image is None:
        gr.Warning("Please upload a garment image.")
        return []

    result = pipeline(
        person_image=person_image,
        garment_image=garment_image,
        category=category,
        garment_photo_type=garment_photo_type,
        segmentation_free=segmentation_free,
        num_timesteps=int(num_timesteps),
        guidance_scale=guidance_scale,
        num_samples=int(num_samples),
        seed=int(seed),
    )
    return result.images


def build_ui() -> gr.Blocks:
    examples_dir = PROJECT_ROOT / "examples" / "data"

    with gr.Blocks(title="FASHN VTON v1.5") as demo:
        gr.Markdown("# FASHN VTON v1.5 — Virtual Try-On Demo\nUpload a person and garment image, then click **Run Try-On**.")

        with gr.Row():
            with gr.Column(scale=1):
                person_image = gr.Image(type="pil", label="Person Image")
                garment_image = gr.Image(type="pil", label="Garment Image")

                category = gr.Radio(
                    choices=["tops", "bottoms", "one-pieces"],
                    value="tops",
                    label="Category",
                )
                garment_photo_type = gr.Radio(
                    choices=["model", "flat-lay"],
                    value="model",
                    label="Garment Photo Type",
                )
                segmentation_free = gr.Checkbox(value=True, label="Segmentation Free")

                with gr.Accordion("Advanced Settings", open=False):
                    num_timesteps = gr.Slider(10, 50, value=30, step=1, label="Sampling Steps")
                    guidance_scale = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="Guidance Scale")
                    num_samples = gr.Slider(1, 4, value=1, step=1, label="Number of Samples")
                    seed = gr.Number(value=42, label="Seed")

                run_btn = gr.Button("Run Try-On", variant="primary")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Results", columns=2)

        inputs = [
            person_image,
            garment_image,
            category,
            garment_photo_type,
            segmentation_free,
            num_timesteps,
            guidance_scale,
            num_samples,
            seed,
        ]

        run_btn.click(fn=run_tryon, inputs=inputs, outputs=gallery)

        if (examples_dir / "model.webp").exists() and (examples_dir / "garment.webp").exists():
            gr.Examples(
                examples=[
                    [str(examples_dir / "model.webp"), str(examples_dir / "garment.webp"), "tops"],
                ],
                inputs=[person_image, garment_image, category],
                label="Examples",
            )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FASHN VTON v1.5 Gradio Demo")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="Path to model weights directory")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda, mps, cpu). Auto-detect if not set.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = TryOnPipeline(weights_dir=args.weights_dir, device=args.device)
    demo = build_ui()
    demo.launch(share=args.share)

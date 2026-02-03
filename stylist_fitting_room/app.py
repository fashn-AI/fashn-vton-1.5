"""AI Personal Stylist + Virtual Fitting Room - Main Gradio App."""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add current directory and parent src to path for imports
APP_DIR = Path(__file__).parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import gradio as gr
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from config import MAX_SUGGESTIONS
from services.gemini_service import GeminiService
from services.search_service import SearchService
from services.vto_service import VTOService
from utils.image_utils import create_placeholder_image


class StylistApp:
    """Main application class for AI Personal Stylist."""

    def __init__(self):
        """Initialize services."""
        self.gemini_service = None
        self.search_service = None
        self.vto_service = None

        # State
        self.current_suggestions = []
        self.user_profile = {}
        self.query_requirements = {}

    def _init_services(self):
        """Lazy initialize services."""
        if self.gemini_service is None:
            try:
                self.gemini_service = GeminiService()
                logger.info("GeminiService initialized")
            except Exception as e:
                logger.error(f"Failed to initialize GeminiService: {e}")
                raise gr.Error(f"Failed to initialize Gemini: {e}")

        if self.search_service is None:
            try:
                self.search_service = SearchService()
                logger.info("SearchService initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SearchService: {e}")
                raise gr.Error(f"Failed to initialize Search: {e}")

        if self.vto_service is None:
            self.vto_service = VTOService()
            logger.info("VTOService initialized (lazy loaded)")

    def find_outfits(
        self,
        person_image: Optional[Image.Image],
        query: str,
    ) -> Tuple[List[Image.Image], str, List[str], str]:
        """
        Main function to find outfit suggestions.

        Args:
            person_image: User's photo
            query: Fashion request text

        Returns:
            Tuple of (suggestion_images, stylist_explanation, suggestion_urls, status)
        """
        if person_image is None:
            raise gr.Error("Please upload your photo first!")

        if not query or not query.strip():
            raise gr.Error("Please describe what kind of outfit you're looking for!")

        self._init_services()

        try:
            # Step 1: Analyze user image
            logger.info("Analyzing user image...")
            self.user_profile = self.gemini_service.analyze_user_image(person_image)
            logger.info(f"User profile: {self.user_profile}")

            # Step 2: Analyze query
            logger.info("Analyzing query...")
            self.query_requirements = self.gemini_service.analyze_query(query)
            logger.info(f"Query requirements: {self.query_requirements}")

            # Step 3: Generate search keywords
            logger.info("Generating search keywords...")
            keywords_result = self.gemini_service.generate_search_keywords(
                self.user_profile,
                self.query_requirements,
            )
            keywords = keywords_result.get("keywords", ["casual outfit"])
            logger.info(f"Search keywords: {keywords}")

            # Step 4: Search for garments
            logger.info("Searching for garments...")
            search_results = self.search_service.search_with_multiple_keywords(
                keywords,
                results_per_keyword=5,
            )

            if not search_results:
                raise gr.Error("No garments found. Try a different search query.")

            # Step 5: Download images and filter valid results
            logger.info("Downloading garment images...")
            valid_results = []
            for result in search_results:
                if len(valid_results) >= MAX_SUGGESTIONS * 2:
                    break

                image = self.search_service.get_image_for_result(result)
                if image is not None:
                    result["image"] = image
                    valid_results.append(result)

            if not valid_results:
                raise gr.Error("Could not download any garment images. Try again.")

            # Step 6: Let stylist select best options
            logger.info("Selecting best outfits...")
            selection = self.gemini_service.select_outfit(
                valid_results[:MAX_SUGGESTIONS * 2],
                self.user_profile,
                self.query_requirements,
            )

            # Prepare suggestions (top MAX_SUGGESTIONS)
            self.current_suggestions = valid_results[:MAX_SUGGESTIONS]

            # Reorder to put selected item first
            selected_idx = selection.get("selected_index", 0)
            if selected_idx > 0 and selected_idx < len(self.current_suggestions):
                self.current_suggestions[0], self.current_suggestions[selected_idx] = (
                    self.current_suggestions[selected_idx],
                    self.current_suggestions[0],
                )

            suggestion_images = [s.get("image") for s in self.current_suggestions]
            suggestion_urls = [s.get("url", "") for s in self.current_suggestions]

            # Format stylist explanation
            explanation = selection.get("explanation", "")
            tips = selection.get("styling_tips", "")
            stylist_text = f"**Why this works for you:** {explanation}\n\n**Styling tips:** {tips}"

            status = f"Found {len(self.current_suggestions)} suggestions based on your style profile!"

            return suggestion_images, stylist_text, suggestion_urls, status

        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Error finding outfits: {e}", exc_info=True)
            raise gr.Error(f"Error finding outfits: {str(e)}")

    def try_on_garment(
        self,
        person_image: Optional[Image.Image],
        garment_index: int,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Run virtual try-on for selected garment.

        Args:
            person_image: User's photo
            garment_index: Index of selected garment

        Returns:
            Tuple of (result_image, status_message)
        """
        if person_image is None:
            raise gr.Error("Please upload your photo first!")

        if not self.current_suggestions:
            raise gr.Error("Please search for outfits first!")

        if garment_index >= len(self.current_suggestions):
            raise gr.Error("Invalid garment selection!")

        self._init_services()

        try:
            garment = self.current_suggestions[garment_index]
            garment_image = garment.get("image")

            if garment_image is None:
                raise gr.Error("Garment image not available!")

            logger.info(f"Running try-on for garment {garment_index}...")

            result_image = self.vto_service.try_on(
                person_image=person_image,
                garment_image=garment_image,
            )

            status = f"Try-on complete! Showing: {garment.get('title', 'Selected garment')}"

            return result_image, status

        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Error in try-on: {e}", exc_info=True)
            raise gr.Error(f"Error in try-on: {str(e)}")


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    app = StylistApp()

    # Create placeholder images
    placeholder = create_placeholder_image(text="Upload your photo")
    garment_placeholder = create_placeholder_image(256, 256, text="Suggestion")

    with gr.Blocks(
        title="AI Personal Stylist + Virtual Fitting Room",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # AI Personal Stylist + Virtual Fitting Room

            Upload your photo, describe what you're looking for, and get personalized outfit suggestions
            that you can virtually try on!
            """
        )

        # State for storing suggestion URLs
        suggestion_urls = gr.State([])

        with gr.Row():
            # Left column: User input
            with gr.Column(scale=1):
                person_image = gr.Image(
                    label="Your Photo",
                    type="pil",
                    height=400,
                )

                query = gr.Textbox(
                    label="What are you looking for?",
                    placeholder="e.g., A casual summer outfit for a beach vacation",
                    lines=2,
                )

                find_btn = gr.Button("Find Outfits", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)

            # Right column: Results
            with gr.Column(scale=2):
                gr.Markdown("### Stylist Suggestions")

                with gr.Row():
                    suggestion_images = []
                    try_buttons = []

                    for i in range(MAX_SUGGESTIONS):
                        with gr.Column(scale=1, min_width=150):
                            img = gr.Image(
                                label=f"Option {i + 1}",
                                type="pil",
                                height=200,
                                interactive=False,
                            )
                            suggestion_images.append(img)

                            btn = gr.Button(f"Try On", size="sm")
                            try_buttons.append(btn)

                stylist_explanation = gr.Markdown(
                    label="Stylist Notes",
                    value="*Search for outfits to see personalized recommendations*",
                )

                gr.Markdown("### Virtual Try-On Result")

                with gr.Row():
                    with gr.Column(scale=2):
                        vto_result = gr.Image(
                            label="Try-On Result",
                            type="pil",
                            height=500,
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        vto_status = gr.Textbox(label="Result", interactive=False)

                        buy_btn = gr.Button(
                            "Buy Now",
                            variant="secondary",
                            size="lg",
                            link="",
                        )

        # Event handlers
        def on_find_outfits(person_img, query_text):
            images, explanation, urls, status = app.find_outfits(person_img, query_text)

            # Pad images list to MAX_SUGGESTIONS
            while len(images) < MAX_SUGGESTIONS:
                images.append(None)

            return (*images, explanation, urls, status)

        find_btn.click(
            fn=on_find_outfits,
            inputs=[person_image, query],
            outputs=[*suggestion_images, stylist_explanation, suggestion_urls, status_text],
        )

        # Create try-on handlers for each button
        def create_try_on_handler(index: int):
            def handler(person_img, urls):
                result_img, status = app.try_on_garment(person_img, index)
                url = urls[index] if index < len(urls) else ""
                return result_img, status, gr.Button(link=url if url else None)

            return handler

        for i, btn in enumerate(try_buttons):
            handler = create_try_on_handler(i)
            btn.click(
                fn=handler,
                inputs=[person_image, suggestion_urls],
                outputs=[vto_result, vto_status, buy_btn],
            )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Upload a full-body photo for best results
            - Be specific about the occasion and style you want
            - Click "Try On" to see how each suggestion looks on you

            *Powered by Gemini 2.0 Flash, Tavily Search, and FASHN VTON 1.5*
            """
        )

    return demo


def main():
    """Main entry point."""
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set. Please set it before using the app.")

    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY not set. Please set it before using the app.")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()

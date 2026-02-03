"""AI Personal Stylist + Virtual Fitting Room - Main Gradio App."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

from config import MAX_TOPS_SUGGESTIONS, MAX_BOTTOMS_SUGGESTIONS, MAX_FULL_SETS
from services.gemini_service import GeminiService
from services.search_service import SearchService
from services.vto_service import VTOService


class StylistApp:
    """Main application class for AI Personal Stylist."""

    def __init__(self):
        """Initialize services."""
        self.gemini_service = None
        self.search_service = None
        self.vto_service = None

        # State
        self.current_tops = []
        self.current_bottoms = []
        self.current_outfit_sets = []
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
        gender: str,
        body_shape: str,
        skin_tone: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Main function to find outfit suggestions with separate tops and bottoms.

        Args:
            person_image: User's photo
            gender: User's gender (male/female/other)
            body_shape: User's body shape
            skin_tone: User's skin tone
            query: Fashion request text

        Returns:
            Dict containing tops, bottoms, outfit_sets, explanation, and status
        """
        if person_image is None:
            raise gr.Error("Please upload your photo first!")

        if not query or not query.strip():
            raise gr.Error("Please describe what kind of outfit you're looking for!")

        self._init_services()

        try:
            # Build user profile from inputs (not auto-detect)
            self.user_profile = {
                "gender": gender,
                "body_shape": body_shape,
                "skin_tone": skin_tone,
            }
            logger.info(f"User profile: {self.user_profile}")

            # Step 1: Analyze query
            logger.info("Analyzing query...")
            self.query_requirements = self.gemini_service.analyze_query(query)
            logger.info(f"Query requirements: {self.query_requirements}")

            # Step 2: Generate separate search keywords for tops and bottoms
            logger.info("Generating search keywords...")
            keywords_result = self.gemini_service.generate_search_keywords(
                self.user_profile,
                self.query_requirements,
            )
            tops_keywords = keywords_result.get("tops_keywords", ["casual shirt"])
            bottoms_keywords = keywords_result.get("bottoms_keywords", ["casual pants"])
            logger.info(f"Tops keywords: {tops_keywords}")
            logger.info(f"Bottoms keywords: {bottoms_keywords}")

            # Step 3: Search for tops
            logger.info("Searching for tops...")
            tops_results = self.search_service.search_tops(
                tops_keywords,
                results_per_keyword=3,
            )

            # Step 4: Search for bottoms
            logger.info("Searching for bottoms...")
            bottoms_results = self.search_service.search_bottoms(
                bottoms_keywords,
                results_per_keyword=3,
            )

            # Step 5: Download images and filter valid results for tops
            logger.info("Downloading top images...")
            valid_tops = []
            for result in tops_results:
                if len(valid_tops) >= MAX_TOPS_SUGGESTIONS:
                    break
                image = self.search_service.get_image_for_result(result)
                if image is not None:
                    result["image"] = image
                    valid_tops.append(result)

            # Step 6: Download images and filter valid results for bottoms
            logger.info("Downloading bottom images...")
            valid_bottoms = []
            for result in bottoms_results:
                if len(valid_bottoms) >= MAX_BOTTOMS_SUGGESTIONS:
                    break
                image = self.search_service.get_image_for_result(result)
                if image is not None:
                    result["image"] = image
                    valid_bottoms.append(result)

            if not valid_tops and not valid_bottoms:
                raise gr.Error("Could not find any garment images. Try a different search query.")

            self.current_tops = valid_tops
            self.current_bottoms = valid_bottoms

            # Step 7: Get outfit set recommendations
            logger.info("Recommending outfit sets...")
            outfit_sets_result = {}
            if valid_tops and valid_bottoms:
                outfit_sets_result = self.gemini_service.recommend_outfit_sets(
                    valid_tops,
                    valid_bottoms,
                    self.user_profile,
                    self.query_requirements,
                    num_sets=MAX_FULL_SETS,
                )
                self.current_outfit_sets = outfit_sets_result.get("outfit_sets", [])
            else:
                self.current_outfit_sets = []

            # Prepare response
            top_images = [t.get("image") for t in valid_tops]
            top_urls = [t.get("url", "") for t in valid_tops]

            bottom_images = [b.get("image") for b in valid_bottoms]
            bottom_urls = [b.get("url", "") for b in valid_bottoms]

            # Format explanation
            reasoning = keywords_result.get("reasoning", "")
            recommended_colors = keywords_result.get("recommended_colors", [])
            styling_tips = outfit_sets_result.get("overall_styling_tips", "")

            explanation = f"**For your {body_shape} figure and {skin_tone} skin tone:**\n\n"
            if reasoning:
                explanation += f"{reasoning}\n\n"
            if recommended_colors:
                explanation += f"**Recommended colors:** {', '.join(recommended_colors)}\n\n"
            if styling_tips:
                explanation += f"**Styling tips:** {styling_tips}"

            status = f"Found {len(valid_tops)} tops and {len(valid_bottoms)} bottoms!"

            return {
                "top_images": top_images,
                "top_urls": top_urls,
                "bottom_images": bottom_images,
                "bottom_urls": bottom_urls,
                "outfit_sets": self.current_outfit_sets,
                "explanation": explanation,
                "status": status,
            }

        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Error finding outfits: {e}", exc_info=True)
            raise gr.Error(f"Error finding outfits: {str(e)}")

    def try_on_garment(
        self,
        person_image: Optional[Image.Image],
        garment_type: str,  # "top" or "bottom"
        garment_index: int,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Run virtual try-on for selected garment.

        Args:
            person_image: User's photo
            garment_type: "top" or "bottom"
            garment_index: Index of selected garment

        Returns:
            Tuple of (result_image, status_message)
        """
        if person_image is None:
            raise gr.Error("Please upload your photo first!")

        garments = self.current_tops if garment_type == "top" else self.current_bottoms
        category = "tops" if garment_type == "top" else "bottoms"

        if not garments:
            raise gr.Error("Please search for outfits first!")

        if garment_index >= len(garments):
            raise gr.Error("Invalid garment selection!")

        self._init_services()

        try:
            garment = garments[garment_index]
            garment_image = garment.get("image")

            if garment_image is None:
                raise gr.Error("Garment image not available!")

            logger.info(f"Running try-on for {garment_type} {garment_index} with category={category}...")

            result_image = self.vto_service.try_on(
                person_image=person_image,
                garment_image=garment_image,
                category=category,  # Explicitly set category
            )

            status = f"Try-on complete! Showing: {garment.get('title', 'Selected garment')}"

            return result_image, status

        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Error in try-on: {e}", exc_info=True)
            raise gr.Error(f"Error in try-on: {str(e)}")

    def try_on_full_set(
        self,
        person_image: Optional[Image.Image],
        set_index: int,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Run virtual try-on for a full outfit set (top first, then bottom).

        Args:
            person_image: User's photo
            set_index: Index of the outfit set

        Returns:
            Tuple of (result_image, status_message)
        """
        if person_image is None:
            raise gr.Error("Please upload your photo first!")

        if not self.current_outfit_sets:
            raise gr.Error("No outfit sets available!")

        if set_index >= len(self.current_outfit_sets):
            raise gr.Error("Invalid outfit set selection!")

        outfit_set = self.current_outfit_sets[set_index]
        top_index = outfit_set.get("top_index", 0)
        bottom_index = outfit_set.get("bottom_index", 0)

        if top_index >= len(self.current_tops) or bottom_index >= len(self.current_bottoms):
            raise gr.Error("Invalid outfit set indices!")

        self._init_services()

        try:
            top = self.current_tops[top_index]
            bottom = self.current_bottoms[bottom_index]

            top_image = top.get("image")
            bottom_image = bottom.get("image")

            if top_image is None or bottom_image is None:
                raise gr.Error("Garment images not available!")

            # Step 1: Try on the top
            logger.info(f"Running try-on for top (set {set_index})...")
            result_with_top = self.vto_service.try_on(
                person_image=person_image,
                garment_image=top_image,
                category="tops",
            )

            # Step 2: Try on the bottom on the result
            logger.info(f"Running try-on for bottom on result (set {set_index})...")
            final_result = self.vto_service.try_on(
                person_image=result_with_top,
                garment_image=bottom_image,
                category="bottoms",
            )

            status = f"Full set try-on complete! {top.get('title', 'Top')} + {bottom.get('title', 'Bottom')}"

            return final_result, status

        except gr.Error:
            raise
        except Exception as e:
            logger.error(f"Error in full set try-on: {e}", exc_info=True)
            raise gr.Error(f"Error in try-on: {str(e)}")


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    app = StylistApp()

    with gr.Blocks(
        title="AI Personal Stylist + Virtual Fitting Room",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # AI Personal Stylist + Virtual Fitting Room

            Upload your photo, tell us about yourself, and get personalized outfit suggestions
            with separate tops and bottoms that you can virtually try on!
            """
        )

        # State variables
        top_urls_state = gr.State([])
        bottom_urls_state = gr.State([])
        outfit_sets_state = gr.State([])

        with gr.Row():
            # Left column: User input
            with gr.Column(scale=1):
                person_image = gr.Image(
                    label="Your Photo",
                    type="pil",
                    height=350,
                )

                gr.Markdown("### Your Profile")
                with gr.Row():
                    gender = gr.Dropdown(
                        label="Gender",
                        choices=["male", "female", "other"],
                        value="male",
                    )
                    body_shape = gr.Dropdown(
                        label="Body Shape",
                        choices=["slim", "average", "athletic", "plus-size"],
                        value="average",
                    )
                    skin_tone = gr.Dropdown(
                        label="Skin Tone",
                        choices=["fair", "medium", "tan", "dark"],
                        value="medium",
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
                # Tops section
                gr.Markdown("### 👕 Top Suggestions")
                with gr.Row():
                    top_images = []
                    top_try_buttons = []

                    for i in range(MAX_TOPS_SUGGESTIONS):
                        with gr.Column(scale=1, min_width=120):
                            img = gr.Image(
                                label=f"Top {i + 1}",
                                type="pil",
                                height=180,
                                interactive=False,
                            )
                            top_images.append(img)

                            btn = gr.Button(f"Try On", size="sm")
                            top_try_buttons.append(btn)

                # Bottoms section
                gr.Markdown("### 👖 Bottom Suggestions")
                with gr.Row():
                    bottom_images = []
                    bottom_try_buttons = []

                    for i in range(MAX_BOTTOMS_SUGGESTIONS):
                        with gr.Column(scale=1, min_width=120):
                            img = gr.Image(
                                label=f"Bottom {i + 1}",
                                type="pil",
                                height=180,
                                interactive=False,
                            )
                            bottom_images.append(img)

                            btn = gr.Button(f"Try On", size="sm")
                            bottom_try_buttons.append(btn)

                # Full Sets section
                gr.Markdown("### 👔👖 Recommended Full Sets")
                with gr.Row():
                    set_top_images = []
                    set_bottom_images = []
                    set_reasons = []
                    set_try_buttons = []

                    for i in range(MAX_FULL_SETS):
                        with gr.Column(scale=1, min_width=200):
                            with gr.Row():
                                set_top_img = gr.Image(
                                    label=f"Set {i + 1} - Top",
                                    type="pil",
                                    height=120,
                                    interactive=False,
                                )
                                set_bottom_img = gr.Image(
                                    label=f"Set {i + 1} - Bottom",
                                    type="pil",
                                    height=120,
                                    interactive=False,
                                )
                            set_top_images.append(set_top_img)
                            set_bottom_images.append(set_bottom_img)

                            set_reason = gr.Markdown(value=f"*Set {i + 1} reasoning will appear here*")
                            set_reasons.append(set_reason)

                            btn = gr.Button(f"Try Full Set {i + 1}", size="sm", variant="secondary")
                            set_try_buttons.append(btn)

                stylist_explanation = gr.Markdown(
                    label="Stylist Notes",
                    value="*Search for outfits to see personalized recommendations*",
                )

        # Virtual Try-On Result section
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
        def on_find_outfits(person_img, gender_val, body_shape_val, skin_tone_val, query_text):
            result = app.find_outfits(person_img, gender_val, body_shape_val, skin_tone_val, query_text)

            # Prepare top images (pad to MAX_TOPS_SUGGESTIONS)
            top_imgs = result["top_images"]
            while len(top_imgs) < MAX_TOPS_SUGGESTIONS:
                top_imgs.append(None)

            # Prepare bottom images (pad to MAX_BOTTOMS_SUGGESTIONS)
            bottom_imgs = result["bottom_images"]
            while len(bottom_imgs) < MAX_BOTTOMS_SUGGESTIONS:
                bottom_imgs.append(None)

            # Prepare outfit set displays
            outfit_sets = result["outfit_sets"]
            set_outputs = []

            for i in range(MAX_FULL_SETS):
                if i < len(outfit_sets):
                    outfit = outfit_sets[i]
                    top_idx = outfit.get("top_index", 0)
                    bottom_idx = outfit.get("bottom_index", 0)

                    # Get images for the set
                    set_top = result["top_images"][top_idx] if top_idx < len(result["top_images"]) else None
                    set_bottom = result["bottom_images"][bottom_idx] if bottom_idx < len(result["bottom_images"]) else None
                    reasoning = outfit.get("reasoning", "Great combination!")

                    set_outputs.extend([set_top, set_bottom, f"**Set {i + 1}:** {reasoning}"])
                else:
                    set_outputs.extend([None, None, f"*Set {i + 1} not available*"])

            return (
                *top_imgs,
                *bottom_imgs,
                *set_outputs,
                result["explanation"],
                result["top_urls"],
                result["bottom_urls"],
                outfit_sets,
                result["status"],
            )

        # Build outputs for set displays
        set_outputs = []
        for i in range(MAX_FULL_SETS):
            set_outputs.extend([set_top_images[i], set_bottom_images[i], set_reasons[i]])

        find_btn.click(
            fn=on_find_outfits,
            inputs=[person_image, gender, body_shape, skin_tone, query],
            outputs=[
                *top_images,
                *bottom_images,
                *set_outputs,
                stylist_explanation,
                top_urls_state,
                bottom_urls_state,
                outfit_sets_state,
                status_text,
            ],
        )

        # Create try-on handlers for top buttons
        def create_top_try_handler(index: int):
            def handler(person_img, top_urls):
                result_img, status = app.try_on_garment(person_img, "top", index)
                url = top_urls[index] if index < len(top_urls) else ""
                return result_img, status, gr.Button(link=url if url else None)

            return handler

        for i, btn in enumerate(top_try_buttons):
            handler = create_top_try_handler(i)
            btn.click(
                fn=handler,
                inputs=[person_image, top_urls_state],
                outputs=[vto_result, vto_status, buy_btn],
            )

        # Create try-on handlers for bottom buttons
        def create_bottom_try_handler(index: int):
            def handler(person_img, bottom_urls):
                result_img, status = app.try_on_garment(person_img, "bottom", index)
                url = bottom_urls[index] if index < len(bottom_urls) else ""
                return result_img, status, gr.Button(link=url if url else None)

            return handler

        for i, btn in enumerate(bottom_try_buttons):
            handler = create_bottom_try_handler(i)
            btn.click(
                fn=handler,
                inputs=[person_image, bottom_urls_state],
                outputs=[vto_result, vto_status, buy_btn],
            )

        # Create try-on handlers for full set buttons
        def create_set_try_handler(index: int):
            def handler(person_img):
                result_img, status = app.try_on_full_set(person_img, index)
                return result_img, status

            return handler

        for i, btn in enumerate(set_try_buttons):
            handler = create_set_try_handler(i)
            btn.click(
                fn=handler,
                inputs=[person_image],
                outputs=[vto_result, vto_status],
            )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Upload a full-body photo for best results
            - Select your profile info for accurate recommendations
            - Be specific about the occasion and style you want
            - Click "Try On" to see how each suggestion looks on you
            - Use "Try Full Set" to try on both top and bottom together

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

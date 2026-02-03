"""Gemini API service for AI-powered fashion analysis."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS
from prompts.analyzer import (
    USER_ANALYSIS_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    SEARCH_KEYWORDS_PROMPT,
    GARMENT_CLASSIFICATION_PROMPT,
)
from prompts.stylist import OUTFIT_SELECTION_PROMPT, STYLIST_PROMPT

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Gemini API for fashion analysis."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini service.

        Args:
            api_key: Gemini API key. If None, uses GEMINI_API_KEY from config.
        """
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set the environment variable.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.generation_config = genai.GenerationConfig(
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_TOKENS,
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            # Remove opening ```json or ```
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            # Remove closing ```
            text = re.sub(r"\n?```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nText: {text}")
            return {}

    def analyze_user_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Analyze a user's photo to extract body shape, skin tone, and gender.

        Args:
            image: PIL Image of the user

        Returns:
            Dict with body_shape, skin_tone, gender, current_style
        """
        try:
            response = self.model.generate_content(
                [USER_ANALYSIS_PROMPT, image],
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            # Validate required fields
            defaults = {
                "body_shape": "average",
                "skin_tone": "medium",
                "gender": "neutral",
                "current_style": "casual",
            }

            for key, default in defaults.items():
                if key not in result:
                    result[key] = default

            return result

        except Exception as e:
            logger.error(f"Error analyzing user image: {e}")
            return {
                "body_shape": "average",
                "skin_tone": "medium",
                "gender": "neutral",
                "current_style": "casual",
            }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user's fashion query to extract requirements.

        Args:
            query: User's text query about what they want to wear

        Returns:
            Dict with style, occasion, weather, items, colors, budget
        """
        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            # Validate required fields
            defaults = {
                "style": "casual",
                "occasion": "daily",
                "weather": "not specified",
                "items": [],
                "colors": [],
                "budget": "not specified",
            }

            for key, default in defaults.items():
                if key not in result:
                    result[key] = default

            return result

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                "style": "casual",
                "occasion": "daily",
                "weather": "not specified",
                "items": [],
                "colors": [],
                "budget": "not specified",
            }

    def generate_search_keywords(
        self,
        user_analysis: Dict[str, Any],
        query_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate search keywords based on user profile and requirements.

        Args:
            user_analysis: Result from analyze_user_image
            query_analysis: Result from analyze_query

        Returns:
            Dict with keywords list, recommended_colors, reasoning
        """
        try:
            prompt = SEARCH_KEYWORDS_PROMPT.format(
                body_shape=user_analysis.get("body_shape", "average"),
                skin_tone=user_analysis.get("skin_tone", "medium"),
                gender=user_analysis.get("gender", "neutral"),
                style=query_analysis.get("style", "casual"),
                occasion=query_analysis.get("occasion", "daily"),
                weather=query_analysis.get("weather", "not specified"),
                items=", ".join(query_analysis.get("items", [])) or "any",
                colors=", ".join(query_analysis.get("colors", [])) or "any",
            )

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            if "keywords" not in result or not result["keywords"]:
                # Generate default keywords based on inputs
                gender = user_analysis.get("gender", "")
                style = query_analysis.get("style", "casual")
                occasion = query_analysis.get("occasion", "daily")
                items = query_analysis.get("items", [])

                default_keywords = []
                if items:
                    for item in items[:3]:
                        default_keywords.append(f"{gender} {style} {item}")
                else:
                    default_keywords = [
                        f"{gender} {style} outfit",
                        f"{gender} {occasion} wear",
                    ]

                result["keywords"] = default_keywords

            return result

        except Exception as e:
            logger.error(f"Error generating search keywords: {e}")
            return {
                "keywords": ["casual outfit", "everyday wear"],
                "recommended_colors": [],
                "reasoning": "Default keywords due to error",
            }

    def select_outfit(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Select the best outfit from candidates based on user profile and requirements.

        Args:
            candidates: List of garment options with title, description, image_url
            user_profile: User analysis results
            requirements: Query analysis results

        Returns:
            Dict with selected_index, explanation, styling_tips
        """
        try:
            # Format candidates for the prompt
            candidates_text = ""
            for i, candidate in enumerate(candidates):
                candidates_text += f"\nOption {i}: {candidate.get('title', 'Unknown')}"
                if candidate.get("description"):
                    candidates_text += f" - {candidate['description']}"
                if candidate.get("price"):
                    candidates_text += f" (${candidate['price']})"

            prompt = OUTFIT_SELECTION_PROMPT.format(
                body_shape=user_profile.get("body_shape", "average"),
                skin_tone=user_profile.get("skin_tone", "medium"),
                gender=user_profile.get("gender", "neutral"),
                current_style=user_profile.get("current_style", "casual"),
                style=requirements.get("style", "casual"),
                occasion=requirements.get("occasion", "daily"),
                weather=requirements.get("weather", "not specified"),
                colors=", ".join(requirements.get("colors", [])) or "any",
                items=", ".join(requirements.get("items", [])) or "any",
                candidates=candidates_text,
            )

            response = self.model.generate_content(
                [STYLIST_PROMPT, prompt],
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            # Validate selected_index
            if "selected_index" not in result or not isinstance(result["selected_index"], int):
                result["selected_index"] = 0
            elif result["selected_index"] >= len(candidates):
                result["selected_index"] = 0

            if "explanation" not in result:
                result["explanation"] = "This option best matches your style preferences."

            if "styling_tips" not in result:
                result["styling_tips"] = "Complete the look with neutral accessories."

            return result

        except Exception as e:
            logger.error(f"Error selecting outfit: {e}")
            return {
                "selected_index": 0,
                "explanation": "Unable to analyze options. Showing first result.",
                "styling_tips": "Try pairing with classic accessories.",
            }

    def classify_garment(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify a garment image to determine category and photo type.

        Args:
            image: PIL Image of the garment

        Returns:
            Dict with category (tops/bottoms/one-pieces), photo_type (model/flat-lay)
        """
        try:
            response = self.model.generate_content(
                [GARMENT_CLASSIFICATION_PROMPT, image],
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            # Validate category
            valid_categories = ["tops", "bottoms", "one-pieces"]
            if result.get("category") not in valid_categories:
                result["category"] = "tops"  # Default to tops

            # Validate photo_type
            valid_photo_types = ["model", "flat-lay"]
            if result.get("photo_type") not in valid_photo_types:
                result["photo_type"] = "model"  # Default to model

            return result

        except Exception as e:
            logger.error(f"Error classifying garment: {e}")
            return {
                "category": "tops",
                "photo_type": "model",
                "description": "Unable to classify garment",
            }

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
from prompts.stylist import OUTFIT_SELECTION_PROMPT, STYLIST_PROMPT, OUTFIT_PAIRING_PROMPT

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
        user_profile: Dict[str, Any],
        query_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate SEPARATE search keywords for tops and bottoms.

        Args:
            user_profile: User profile with gender, body_shape, skin_tone
            query_analysis: Result from analyze_query

        Returns:
            Dict with tops_keywords, bottoms_keywords, recommended_colors, reasoning
        """
        try:
            prompt = SEARCH_KEYWORDS_PROMPT.format(
                body_shape=user_profile.get("body_shape", "average"),
                skin_tone=user_profile.get("skin_tone", "medium"),
                gender=user_profile.get("gender", "neutral"),
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

            # Ensure we have separate top and bottom keywords
            gender = user_profile.get("gender", "")
            gender_prefix = "men" if gender == "male" else ("women" if gender == "female" else "unisex")
            style = query_analysis.get("style", "casual")
            occasion = query_analysis.get("occasion", "daily")

            if "tops_keywords" not in result or not result["tops_keywords"]:
                result["tops_keywords"] = [
                    f"{gender_prefix} {style} shirt",
                    f"{gender_prefix} {occasion} t-shirt",
                    f"{gender_prefix} casual top",
                ]

            if "bottoms_keywords" not in result or not result["bottoms_keywords"]:
                result["bottoms_keywords"] = [
                    f"{gender_prefix} {style} pants",
                    f"{gender_prefix} {occasion} jeans",
                    f"{gender_prefix} casual shorts",
                ]

            # Legacy support: also provide combined keywords
            if "keywords" not in result:
                result["keywords"] = result["tops_keywords"] + result["bottoms_keywords"]

            return result

        except Exception as e:
            logger.error(f"Error generating search keywords: {e}")
            return {
                "tops_keywords": ["casual shirt", "everyday t-shirt"],
                "bottoms_keywords": ["casual pants", "everyday jeans"],
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

    def recommend_outfit_sets(
        self,
        tops: List[Dict[str, Any]],
        bottoms: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        requirements: Dict[str, Any],
        num_sets: int = 2,
    ) -> Dict[str, Any]:
        """
        Recommend outfit sets by pairing tops with bottoms.

        Args:
            tops: List of top garment options with title, description
            bottoms: List of bottom garment options with title, description
            user_profile: User profile data
            requirements: Style requirements from query analysis
            num_sets: Number of outfit sets to recommend

        Returns:
            Dict with outfit_sets and overall_styling_tips
        """
        try:
            # Format tops list
            tops_text = ""
            for i, top in enumerate(tops):
                tops_text += f"\n{i}: {top.get('title', 'Unknown top')}"
                if top.get("description"):
                    tops_text += f" - {top['description'][:100]}"

            # Format bottoms list
            bottoms_text = ""
            for i, bottom in enumerate(bottoms):
                bottoms_text += f"\n{i}: {bottom.get('title', 'Unknown bottom')}"
                if bottom.get("description"):
                    bottoms_text += f" - {bottom['description'][:100]}"

            prompt = OUTFIT_PAIRING_PROMPT.format(
                body_shape=user_profile.get("body_shape", "average"),
                skin_tone=user_profile.get("skin_tone", "medium"),
                gender=user_profile.get("gender", "neutral"),
                style=requirements.get("style", "casual"),
                occasion=requirements.get("occasion", "daily"),
                tops_list=tops_text,
                bottoms_list=bottoms_text,
                num_sets=num_sets,
            )

            response = self.model.generate_content(
                [STYLIST_PROMPT, prompt],
                generation_config=self.generation_config,
            )
            result = self._extract_json(response.text)

            # Validate outfit_sets
            if "outfit_sets" not in result or not result["outfit_sets"]:
                # Create default pairings
                result["outfit_sets"] = []
                for i in range(min(num_sets, min(len(tops), len(bottoms)))):
                    result["outfit_sets"].append({
                        "top_index": i,
                        "bottom_index": i,
                        "reasoning": "Default pairing based on order.",
                    })

            # Validate indices
            valid_sets = []
            for outfit in result["outfit_sets"]:
                top_idx = outfit.get("top_index", 0)
                bottom_idx = outfit.get("bottom_index", 0)
                if 0 <= top_idx < len(tops) and 0 <= bottom_idx < len(bottoms):
                    valid_sets.append(outfit)

            result["outfit_sets"] = valid_sets[:num_sets]

            if "overall_styling_tips" not in result:
                result["overall_styling_tips"] = "Complete the look with matching accessories."

            return result

        except Exception as e:
            logger.error(f"Error recommending outfit sets: {e}")
            return {
                "outfit_sets": [
                    {"top_index": 0, "bottom_index": 0, "reasoning": "Default pairing."}
                ],
                "overall_styling_tips": "Try neutral accessories to complete the look.",
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

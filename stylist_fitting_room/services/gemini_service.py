"""Gemini API service for AI-powered fashion analysis."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel, Field
import tenacity

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS


# ============================================================================
# Pydantic Response Models for Gemini Structured Output
# ============================================================================


class UserAnalysisResponse(BaseModel):
    """Response schema for user image analysis."""

    body_shape: str = Field(
        default="average",
        description="Body shape: slim/average/athletic/curvy/plus-size",
    )
    skin_tone: str = Field(
        default="medium",
        description="Skin tone: fair/light/medium/olive/tan/dark",
    )
    gender: str = Field(
        default="neutral",
        description="Gender presentation: male/female/neutral",
    )
    current_style: str = Field(
        default="casual",
        description="Current style observed: casual/formal/sporty/etc.",
    )


class QueryAnalysisResponse(BaseModel):
    """Response schema for query analysis."""

    style: str = Field(
        default="casual",
        description="Style: casual/formal/vintage/streetwear/minimalist/athletic/beach/other",
    )
    occasion: str = Field(
        default="daily",
        description="Occasion: work/party/date/travel/daily/wedding/beach/other",
    )
    weather: str = Field(
        default="not specified",
        description="Weather: hot/cold/mild/not specified",
    )
    items: List[str] = Field(
        default_factory=list,
        description="Specific items mentioned (e.g., dress, jeans, blazer)",
    )
    colors: List[str] = Field(
        default_factory=list,
        description="Colors mentioned or preferred",
    )
    budget: str = Field(
        default="not specified",
        description="Budget: low/medium/high/not specified",
    )


class SearchKeywordsResponse(BaseModel):
    """Response schema for search keywords generation."""

    tops_keywords: List[str] = Field(
        default_factory=list,
        description="2-3 search keywords for tops/shirts/blouses",
    )
    bottoms_keywords: List[str] = Field(
        default_factory=list,
        description="2-3 search keywords for bottoms/pants/skirts",
    )
    recommended_colors: List[str] = Field(
        default_factory=list,
        description="Recommended colors based on user profile",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of keyword choices",
    )


class OutfitSelectionResponse(BaseModel):
    """Response schema for outfit selection."""

    selected_index: int = Field(
        default=0,
        description="Index of the selected outfit from candidates (0-based)",
    )
    explanation: str = Field(
        default="",
        description="Why this outfit was selected",
    )
    styling_tips: str = Field(
        default="",
        description="Tips for styling this outfit",
    )


class OutfitSetItem(BaseModel):
    """A single outfit set pairing a top with a bottom."""

    top_index: int = Field(description="Index of the top garment (0-based)")
    bottom_index: int = Field(description="Index of the bottom garment (0-based)")
    reasoning: str = Field(
        default="",
        description="Why this pairing works well together",
    )


class OutfitSetsResponse(BaseModel):
    """Response schema for outfit set recommendations."""

    outfit_sets: List[OutfitSetItem] = Field(
        default_factory=list,
        description="List of recommended outfit pairings",
    )
    overall_styling_tips: str = Field(
        default="",
        description="General styling tips for all outfits",
    )


class GarmentClassificationResponse(BaseModel):
    """Response schema for garment classification."""

    category: str = Field(
        default="tops",
        description="Garment category: tops/bottoms/one-pieces",
    )
    photo_type: str = Field(
        default="model",
        description="Photo type: model/flat-lay",
    )
    description: str = Field(
        default="",
        description="Brief description of the garment",
    )


# ============================================================================
# Imports for prompts (after model definitions for clarity)
# ============================================================================

from prompts.analyzer import (
    USER_ANALYSIS_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    SEARCH_KEYWORDS_PROMPT,
    GARMENT_CLASSIFICATION_PROMPT,
)
from prompts.stylist import OUTFIT_SELECTION_PROMPT, STYLIST_PROMPT, OUTFIT_PAIRING_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# Helper for retry logic
# ============================================================================

def _create_retry_decorator():
    """Create a retry decorator for Gemini API calls."""
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=4),
        retry=tenacity.retry_if_exception_type((json.JSONDecodeError, ValueError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying Gemini API call, attempt {retry_state.attempt_number}"
        ),
    )


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

    def _get_structured_config(self, response_schema: type) -> genai.GenerationConfig:
        """
        Create a generation config with structured output (JSON schema enforcement).

        Args:
            response_schema: Pydantic model class defining the expected response structure

        Returns:
            GenerationConfig with response_mime_type and response_schema
        """
        return genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_TOKENS,
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response, handling markdown code blocks (legacy fallback)."""
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
                generation_config=self._get_structured_config(UserAnalysisResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            result = UserAnalysisResponse.model_validate_json(response.text)
            return result.model_dump()

        except Exception as e:
            logger.error(f"Error analyzing user image: {e}")
            return UserAnalysisResponse().model_dump()

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
                generation_config=self._get_structured_config(QueryAnalysisResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            result = QueryAnalysisResponse.model_validate_json(response.text)
            return result.model_dump()

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return QueryAnalysisResponse().model_dump()

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
                generation_config=self._get_structured_config(SearchKeywordsResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            parsed = SearchKeywordsResponse.model_validate_json(response.text)
            result = parsed.model_dump()

            # Ensure we have separate top and bottom keywords (fallback if empty)
            gender = user_profile.get("gender", "")
            gender_prefix = "men" if gender == "male" else ("women" if gender == "female" else "unisex")
            style = query_analysis.get("style", "casual")
            occasion = query_analysis.get("occasion", "daily")

            if not result["tops_keywords"]:
                result["tops_keywords"] = [
                    f"{gender_prefix} {style} shirt",
                    f"{gender_prefix} {occasion} t-shirt",
                    f"{gender_prefix} casual top",
                ]

            if not result["bottoms_keywords"]:
                result["bottoms_keywords"] = [
                    f"{gender_prefix} {style} pants",
                    f"{gender_prefix} {occasion} jeans",
                    f"{gender_prefix} casual shorts",
                ]

            # Legacy support: also provide combined keywords
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
                generation_config=self._get_structured_config(OutfitSelectionResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            parsed = OutfitSelectionResponse.model_validate_json(response.text)
            result = parsed.model_dump()

            # Validate selected_index is within bounds
            if result["selected_index"] >= len(candidates):
                result["selected_index"] = 0

            # Provide defaults for empty strings
            if not result["explanation"]:
                result["explanation"] = "This option best matches your style preferences."

            if not result["styling_tips"]:
                result["styling_tips"] = "Complete the look with neutral accessories."

            return result

        except Exception as e:
            logger.error(f"Error selecting outfit: {e}")
            return OutfitSelectionResponse(
                explanation="Unable to analyze options. Showing first result.",
                styling_tips="Try pairing with classic accessories.",
            ).model_dump()

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
                generation_config=self._get_structured_config(OutfitSetsResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            parsed = OutfitSetsResponse.model_validate_json(response.text)
            result = parsed.model_dump()

            # Validate outfit_sets (create defaults if empty)
            if not result["outfit_sets"]:
                result["outfit_sets"] = []
                for i in range(min(num_sets, min(len(tops), len(bottoms)))):
                    result["outfit_sets"].append({
                        "top_index": i,
                        "bottom_index": i,
                        "reasoning": "Default pairing based on order.",
                    })

            # Validate indices and filter invalid ones
            valid_sets = []
            for outfit in result["outfit_sets"]:
                top_idx = outfit.get("top_index", 0)
                bottom_idx = outfit.get("bottom_index", 0)
                if 0 <= top_idx < len(tops) and 0 <= bottom_idx < len(bottoms):
                    valid_sets.append(outfit)

            result["outfit_sets"] = valid_sets[:num_sets]

            # Provide default if empty
            if not result["overall_styling_tips"]:
                result["overall_styling_tips"] = "Complete the look with matching accessories."

            return result

        except Exception as e:
            logger.error(f"Error recommending outfit sets: {e}")
            return OutfitSetsResponse(
                outfit_sets=[OutfitSetItem(top_index=0, bottom_index=0, reasoning="Default pairing.")],
                overall_styling_tips="Try neutral accessories to complete the look.",
            ).model_dump()

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
                generation_config=self._get_structured_config(GarmentClassificationResponse),
            )
            # Parse with Pydantic for guaranteed valid structure
            parsed = GarmentClassificationResponse.model_validate_json(response.text)
            result = parsed.model_dump()

            # Validate category (ensure it's one of the valid values)
            valid_categories = ["tops", "bottoms", "one-pieces"]
            if result["category"] not in valid_categories:
                result["category"] = "tops"

            # Validate photo_type
            valid_photo_types = ["model", "flat-lay"]
            if result["photo_type"] not in valid_photo_types:
                result["photo_type"] = "model"

            return result

        except Exception as e:
            logger.error(f"Error classifying garment: {e}")
            return GarmentClassificationResponse(
                description="Unable to classify garment"
            ).model_dump()

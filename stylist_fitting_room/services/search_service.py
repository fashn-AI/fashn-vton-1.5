"""Tavily search service for finding garments online."""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image
from tavily import TavilyClient

from config import TAVILY_API_KEY, SEARCH_NUM_RESULTS
from utils.image_utils import download_image

logger = logging.getLogger(__name__)


class SearchService:
    """Service for searching garments using Tavily API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Tavily search service.

        Args:
            api_key: Tavily API key. If None, uses TAVILY_API_KEY from config.
        """
        self.api_key = api_key or TAVILY_API_KEY
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not set. Please set the environment variable.")

        self.client = TavilyClient(api_key=self.api_key)
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def search_garments(
        self,
        keywords: str,
        num_results: int = SEARCH_NUM_RESULTS,
        include_images: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for garments using Tavily API.

        Args:
            keywords: Search keywords
            num_results: Number of results to return
            include_images: Whether to include image URLs in results

        Returns:
            List of dicts with title, url, image_url, snippet, price
        """
        # Check cache
        cache_key = f"{keywords}:{num_results}"
        if cache_key in self._cache:
            logger.info(f"Returning cached results for: {keywords}")
            return self._cache[cache_key]

        try:
            # Add shopping-related terms to improve results
            search_query = f"{keywords} buy online shop"

            response = self.client.search(
                query=search_query,
                search_depth="basic",
                max_results=num_results,
                include_images=include_images,
            )

            results = []

            # Process search results
            for result in response.get("results", []):
                item = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "image_url": None,
                    "price": None,
                }

                # Extract price from snippet if present
                price = self._extract_price(item["snippet"])
                if price:
                    item["price"] = price

                results.append(item)

            # Add images from Tavily's image results
            images = response.get("images", [])
            for i, image_url in enumerate(images):
                if i < len(results):
                    results[i]["image_url"] = image_url
                else:
                    # Add extra results for images without matching search results
                    results.append({
                        "title": f"Fashion Item {i + 1}",
                        "url": "",
                        "snippet": "",
                        "image_url": image_url,
                        "price": None,
                    })

            # Cache results
            self._cache[cache_key] = results
            logger.info(f"Found {len(results)} results for: {keywords}")

            return results

        except Exception as e:
            logger.error(f"Error searching for garments: {e}")
            return []

    def search_with_multiple_keywords(
        self,
        keywords_list: List[str],
        results_per_keyword: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search with multiple keyword combinations and combine results.

        Args:
            keywords_list: List of keyword combinations to search
            results_per_keyword: Number of results per keyword

        Returns:
            Combined and deduplicated list of results
        """
        all_results = []
        seen_urls = set()

        for keywords in keywords_list:
            results = self.search_garments(keywords, num_results=results_per_keyword)

            for result in results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)

        return all_results

    def download_garment_image(self, url: str) -> Optional[Image.Image]:
        """
        Download a garment image from URL.

        Args:
            url: Image URL to download

        Returns:
            PIL Image or None if download failed
        """
        return download_image(url)

    def get_image_for_result(self, result: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Get the image for a search result, downloading if necessary.

        Args:
            result: Search result dict with image_url

        Returns:
            PIL Image or None
        """
        image_url = result.get("image_url")
        if not image_url:
            return None

        return self.download_garment_image(image_url)

    def _extract_price(self, text: str) -> Optional[str]:
        """
        Extract price from text snippet.

        Args:
            text: Text that may contain price

        Returns:
            Price string or None
        """
        import re

        # Common price patterns
        patterns = [
            r"\$\d+(?:\.\d{2})?",  # $XX.XX or $XX
            r"USD\s*\d+(?:\.\d{2})?",  # USD XX.XX
            r"\d+(?:\.\d{2})?\s*(?:USD|dollars?)",  # XX USD
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def clear_cache(self):
        """Clear the search results cache."""
        self._cache.clear()
        logger.info("Search cache cleared")

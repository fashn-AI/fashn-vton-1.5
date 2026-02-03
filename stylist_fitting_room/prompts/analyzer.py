"""Analyzer prompts for user and query analysis."""

USER_ANALYSIS_PROMPT = """Analyze this person's photo and extract the following information.
Look carefully at their physical characteristics and any visible clothing.

Extract:
1. Body shape: slim / average / athletic / plus-size
2. Skin tone: fair / medium / tan / dark
3. Apparent gender: male / female / neutral
4. Current outfit style (if visible): describe briefly

Return ONLY a valid JSON object with no additional text:
{
    "body_shape": "...",
    "skin_tone": "...",
    "gender": "...",
    "current_style": "..."
}
"""

QUERY_ANALYSIS_PROMPT = """Analyze this fashion request and extract the key requirements.

User request: {query}

Extract:
1. Desired style: casual / formal / vintage / streetwear / minimalist / bohemian / preppy / athletic / other
2. Occasion: work / party / date / travel / daily / wedding / interview / gym / beach / other
3. Weather hints: hot / cold / mild / rainy / not specified
4. Specific items mentioned: list any specific garment types (e.g., dress, jeans, blazer)
5. Color preferences: any colors mentioned
6. Budget hints: luxury / affordable / not specified

Return ONLY a valid JSON object with no additional text:
{
    "style": "...",
    "occasion": "...",
    "weather": "...",
    "items": [...],
    "colors": [...],
    "budget": "..."
}
"""

SEARCH_KEYWORDS_PROMPT = """Based on the user profile and their fashion request, generate optimal search keywords for finding clothes online.

User Profile:
- Body shape: {body_shape}
- Skin tone: {skin_tone}
- Gender: {gender}

Fashion Request:
- Style: {style}
- Occasion: {occasion}
- Weather: {weather}
- Specific items: {items}
- Color preferences: {colors}

Generate 3-5 search keyword combinations that would find the best matching garments.
Consider colors that complement the user's skin tone.
Consider styles that flatter the user's body shape.

Return ONLY a valid JSON object with no additional text:
{
    "keywords": [
        "keyword combination 1",
        "keyword combination 2",
        ...
    ],
    "recommended_colors": ["color1", "color2"],
    "reasoning": "brief explanation of why these keywords were chosen"
}
"""

GARMENT_CLASSIFICATION_PROMPT = """Analyze this garment image and classify it.

Determine:
1. Category: Is this garment for the upper body (tops), lower body (bottoms), or a full-body piece (one-pieces)?
   - tops: shirts, t-shirts, blouses, sweaters, jackets, coats, vests
   - bottoms: pants, jeans, shorts, skirts
   - one-pieces: dresses, jumpsuits, rompers, overalls

2. Photo type: Is this a model photo (worn by a person) or a flat-lay (product shot on plain background)?
   - model: garment is being worn by a person
   - flat-lay: garment is laid flat or on a mannequin/hanger

Return ONLY a valid JSON object with no additional text:
{
    "category": "tops" | "bottoms" | "one-pieces",
    "photo_type": "model" | "flat-lay",
    "description": "brief description of the garment"
}
"""

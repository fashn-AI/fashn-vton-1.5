"""Configuration for AI Personal Stylist + Virtual Fitting Room."""

import os

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# VTO Settings
VTO_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
VTO_NUM_TIMESTEPS = 20  # Fast inference (20 steps vs default 30)
VTO_GUIDANCE_SCALE = 1.5

# Search Settings
SEARCH_NUM_RESULTS = 10
SEARCH_INCLUDE_IMAGES = True

# Gemini Settings
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.7
GEMINI_MAX_TOKENS = 1024

# UI Settings
MAX_SUGGESTIONS = 4

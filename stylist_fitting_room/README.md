# AI Personal Stylist + Virtual Fitting Room

A Gradio application that combines AI-powered fashion recommendations with virtual try-on capabilities.

**Tech Stack:** Gradio | Gemini 2.0 Flash | Tavily Search | FASHN VTON 1.5

---

## Features

- **AI-Powered Outfit Recommendations** - Personalized suggestions based on user profile (body shape, skin tone, gender)
- **Separate Top & Bottom Suggestions** - Independent garment recommendations for mix-and-match flexibility
- **Outfit Set Pairing** - Curated outfit combinations that work well together
- **Virtual Try-On** - Visualize garments on your photo using FASHN VTON 1.5

---

## Project Structure

```
stylist_fitting_room/
├── app.py                 # Main Gradio application
├── config.py              # Configuration & environment variables
├── requirements.txt       # Dependencies
├── .env.example           # Environment template
├── GITFLOW.md             # Branch strategy documentation
├── services/
│   ├── gemini_service.py  # Gemini AI integration
│   ├── search_service.py  # Tavily search for garments
│   └── vto_service.py     # Virtual try-on wrapper
├── prompts/
│   ├── analyzer.py        # User & query analysis prompts
│   └── stylist.py         # Outfit selection prompts
└── utils/
    └── image_utils.py     # Image download/processing utilities
```

---

## Setup

### Prerequisites

- Python 3.10+
- Parent FASHN VTON installation (for virtual try-on)

### Installation

1. **Install parent project dependencies** (from `fashn-vton-1.5/` root):
   ```bash
   pip install -e .
   ```

2. **Install stylist app dependencies**:
   ```bash
   cd stylist_fitting_room
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   TAVILY_API_KEY=your-tavily-api-key-here
   ```

4. **Run the app**:
   ```bash
   python app.py
   ```

---

## Development Workflow

See [GITFLOW.md](./GITFLOW.md) for detailed branch strategy and commit conventions.

### Quick Commands

```bash
# Start new feature
git checkout main && git pull origin main
git checkout -b feature/your-feature-name

# Commit changes
git add -A && git commit -m "feat(scope): description"

# Push and create PR
git push origin feature/your-feature-name
```

### Commit Convention

```
<type>(<scope>): <description>
```

| Type       | Use Case            |
|------------|---------------------|
| `feat`     | New feature         |
| `fix`      | Bug fix             |
| `docs`     | Documentation       |
| `refactor` | Code restructure    |
| `chore`    | Maintenance         |

| Scope    | Area                    |
|----------|-------------------------|
| `gemini` | Gemini AI service       |
| `vto`    | Virtual try-on pipeline |
| `search` | Tavily search           |
| `app`    | Main Gradio application |

---

## Configuration Reference

| Variable | Purpose | Source |
|----------|---------|--------|
| `GEMINI_API_KEY` | Gemini 2.0 Flash API access | [Google AI Studio](https://aistudio.google.com/) |
| `TAVILY_API_KEY` | Garment image search | [Tavily](https://tavily.com/) |

### VTO Settings (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `VTO_NUM_TIMESTEPS` | 20 | Inference steps (lower = faster) |
| `VTO_GUIDANCE_SCALE` | 1.5 | Generation guidance |
| `SEARCH_NUM_RESULTS` | 10 | Garment search results |
| `MAX_TOPS_SUGGESTIONS` | 4 | Top garment suggestions |
| `MAX_BOTTOMS_SUGGESTIONS` | 4 | Bottom garment suggestions |
| `MAX_FULL_SETS` | 2 | Complete outfit sets |

---

## Architecture Overview

```
User Input (photo + profile + query)
         │
         ▼
   ┌─────────────┐
   │   Gemini    │ ◄── Analyze user request
   │  (Analyzer) │
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   Tavily    │ ◄── Search for garment images
   │   Search    │
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │   Gemini    │ ◄── Select & pair outfits
   │  (Stylist)  │
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ FASHN VTON  │ ◄── Virtual try-on
   │   1.5       │
   └─────────────┘
         │
         ▼
    Try-On Result
```

---

## API Key Sources

- **Gemini API Key**: [Google AI Studio](https://aistudio.google.com/) - Free tier available
- **Tavily API Key**: [Tavily](https://tavily.com/) - Free tier with 1000 searches/month

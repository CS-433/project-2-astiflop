# Project Presentation (Slidev)

Slide deck for the **C. elegans lifespan & Terbinafine** project.

## Quick start

```bash
cd presentation
npm install   # first time only
npm run dev   # opens the slide editor in the browser
```

## Slide structure (13 slides)

1. Title
2. Group presentation
3. Intro — the task
4. Terbinafine classification
5. Preprocessing (animated pipeline)
6. Architecture — feature extraction
7. Architecture — attention aggregation
8. Architecture — BiLSTM & TCN
9. Training — staircase sampling
10. Benchmark results
11. Visualization & interpretability
12. Conclusion
13. Thanks & questions

## Customization

- **Team names:** edit slide 2 in [`slides.md`](./slides.md)
- **Benchmark numbers/plots:** slide 10 — add results from `avg_results.json` or benchmark plots
- **Attention plots:** slide 11 — replace placeholder with output from `visualization_pipeline.py`
- **Styling:** gradient backgrounds cycle in [`styles/index.css`](./styles/index.css)

## Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server with live reload |
| `npm run build` | Build a static SPA |
| `npm run export` | Export slides to PDF |

## Components

- [`PreprocessingFlow.vue`](./components/PreprocessingFlow.vue) — animated video → tensor pipeline
- [`ArchitectureDiagram.vue`](./components/ArchitectureDiagram.vue) — model architecture visuals

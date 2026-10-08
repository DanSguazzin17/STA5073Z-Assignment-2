# STA5073Z Assignment 2 — Handwriting Writer Identification

Image classification with neural networks: identifying handwriting writers using a CNN and a Siamese network.

## What's in this repo

- `data/handwriting.rds` — the assignment dataset (25,000 images, 20 writers, digits 0-4, 5 sessions)
- `R/` — reusable functions (data loading, pair construction, metrics)
- `code/` — model training scripts
- `report.qmd` — the Quarto report (renders to the submitted write-up)
- `_quarto.yml` — website config for GitHub Pages
- `roles.txt` — group member names, roles, and links (submission deliverable)

## How to reproduce

1. Open `STA5073Z_Assignment2.Rproj` in RStudio.
2. Run `renv::restore()` to install the exact package versions this project uses.
3. Render the report: click **Render** on `report.qmd`, or run `quarto render` in the terminal.

## Group members

See `roles.txt`.

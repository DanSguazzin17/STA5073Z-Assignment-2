# STA5073Z Assignment 2 — Handwriting Writer Identification

Image classification with neural networks: identifying handwriting writers using a CNN and a Siamese network.

## What's in this repo

- `data/handwriting.rds` — the assignment dataset (25,000 images, 20 writers, digits 0-4, 5 sessions)
- `R/` — shared code used by all tasks
  - `setup.R` — loads all packages and sources the other files in `R/`
  - `data.R` — data loading and train/validation/test splits
  - `cnn.R` — CNN architecture and training
  - `evaluation.R` — prediction and evaluation helpers
  - `plotting.R` — plot theme and figure saving
- `task_1/`, `task_2/`, ... — one folder per task with its script, figures and results
- `report.qmd` — the Quarto report (renders to the submitted write-up)
- `_quarto.yml` — website config for GitHub Pages
- `roles.txt` — group member names, roles, and links (submission deliverable)

## How to reproduce

1. Open `STA5073Z_Assignment2.Rproj` in RStudio (all paths are resolved from the project root).
2. Run `renv::restore()` to install the exact package versions this project uses.
3. Run `keras3::install_keras()` once per machine to set up the Python backend (TensorFlow).
4. Run the task scripts, e.g. `task_2/cnn_writer_id.R`. Each script starts with
   `source(here::here("R", "setup.R"))`, which loads all packages and shared functions.
5. Render the report: click **Render** on `report.qmd`, or run `quarto render` in the terminal.

## Group members

See `roles.txt`.

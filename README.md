# Writer Identification from Handwritten Digits

STA5073Z Data Science for Industry, Assignment 2 (University of Cape Town, 2026)

We use 25,000 simulated images of handwritten digits (20 writers × digits 0–4 × 5 recording
sessions × 50 replicates) to answer two questions:

1. **Who wrote this digit?** A convolutional neural network (CNN) identifies the writer of a single image.
2. **Were these two digits written by the same person?** A Siamese network compares two images.

| Model | Evaluated on | Result |
|---|---|---|
| CNN | Session 5 (unseen) | 73.3% accuracy, 91.7% top-3 accuracy (chance: 5%) |
| Siamese network | Pairs within session 5 | 98.2% balanced accuracy |
| Siamese network | Pairs across sessions 4 and 5 | 66.6% balanced accuracy |

The full write-up is in [`report.qmd`](report.qmd). Group members and roles are in [`roles.txt`](roles.txt).

---

## Repository structure

```
├── report.qmd               Report (Quarto); only reads saved results, does not retrain models
├── references.bib           Bibliography for the report
├── roles.txt                Group members, roles and links (submission deliverable)
├── data/
│   └── handwriting.rds      The assignment data set
├── R/                       Shared functions, loaded by every task script via R/setup.R
├── task_1/                  Task 1: data exploration
├── task_2/                  Task 2: CNN for writer identification
├── task_3/                  Task 3: Siamese network (incl. its hyperparameter search)
├── task_5/                  Task 5: hyperparameter tuning of the CNN
└── .github/workflows/       CI: checks that all R files parse and that the report renders
```

Each task folder contains its script(s), its figures and, where needed, a `results/` folder with
the saved results that the report reads.

### Shared code (`R/`)

| File | Contents |
|---|---|
| `setup.R` | Loads packages and sources the other files in `R/`. Every task script starts with it. |
| `data.R` | `load_handwriting()`, `assign_session_split()`, `assign_random_split()`, `get_set()` |
| `cnn.R` | `build_cnn()`, `fit_cnn()` (with early stopping), `fit_cnn_final()` |
| `evaluation.R` | `predict_writer()`, `accuracy_by()`, `confusion_table()`, `aggregated_accuracy()` |
| `tuning.R` | `evaluate_config()`, `overall_by_run()` for the CNN tuning |
| `plotting.R` | Colour scheme (`plot_colours`), plot theme and `save_figure()` |
| `01_load_data.R` | Standalone loading script used by the Siamese network (not sourced by `setup.R`) |

---

## Data split

All models use the same split by **recording session**, so that every evaluation is on a session
the model has not seen:

| Set | Sessions | Used for |
|---|---|---|
| Training | 1–3 | Fitting the models |
| Validation | 4 | Early stopping, model selection, hyperparameter tuning |
| Test | 5 | One final evaluation |

The final models are retrained on sessions 1–4 and evaluated once on session 5. A random
image-level split is only used in `task_2/cnn_writer_id.R` to show how much it would overstate
performance.

---

## How to reproduce

### 1. Setup (once per machine)

1. Open `STA5073Z_Assignment2.Rproj` in RStudio. All paths are resolved from the project root.
2. Install the R packages:
   ```r
   install.packages(c("dplyr", "tidyr", "purrr", "ggplot2", "scales", "here", "withr",
                      "keras3", "knitr", "rmarkdown"))
   ```
3. Set up the Python backend for keras3 (TensorFlow):
   ```r
   keras3::install_keras()
   ```
   `R/setup.R` requests Python 3.12, as TensorFlow has no build for newer versions yet.

### 2. Run the task scripts

Run the scripts in this order. Each one saves its figures and results, which later scripts and
the report read.

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `task_1/data_exploration.R` | Example images, feature ANOVA, pixel variance decomposition, simulation artefacts | `task_1/exploration_plots/`, `task_1/results/exploration_results.rds` |
| 2 | `task_2/cnn_writer_id.R` | Baseline CNN; comparison of session split and random split | `task_2/cnn_plots/`, `task_2/results/cnn_results.rds` |
| 3 | `task_2/learning_curves.R` | Learning curves at two learning rates (why early stopping uses validation accuracy) | `task_2/results/learning_curves.rds` |
| 4 | `task_5/cnn_tuning.R` | One-at-a-time tuning of shift, dropout, filters and depth, 3 seeds each (longest step) | `task_5/tuning_plots/`, `task_5/results/tuning_results.rds` |
| 5 | `task_2/cnn_final.R` | Final CNN on sessions 1–4, tested on session 5 with 5 seeds | `task_2/final_plots/`, `task_2/results/cnn_final_results.rds` |
| 6 | `task_3/siamese_network.R` | Siamese network: pair construction, tuning, final model, cross-session check | `task_3/results/` |

Step 5 needs the results of step 4.

### 3. Render the report

Click **Render** on `report.qmd`, or run in the terminal:

```bash
quarto render report.qmd
```

The report only reads the saved results and figures, so it renders in seconds.

---

## Collaboration

- `main` holds the submitted version; `dev` collects finished work.
- Each task was developed on its own branch (`data_exploration`, `cnn`, `siamese-network`) and
  merged into `dev` through pull requests.
- The CI workflow runs on every push and pull request: it checks that all R files parse, loads the
  shared functions and renders the report.

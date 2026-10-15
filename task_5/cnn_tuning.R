# Task 5: Limited hyperparameter optimisation for the CNN
# One-at-a-time design: starting from the Task 2 baseline, each hyperparameter is
# varied while the others stay fixed. Every configuration is trained with several
# seeds, because the seed alone moves validation accuracy by about +-2 points.
# All decisions use the validation session (4); the test session (5) is not touched.

here::i_am("task_5/cnn_tuning.R")
source(here::here("R", "setup.R"))

use_backend("tensorflow")

figure_dir <- here::here("task_5", "tuning_plots")
result_dir <- here::here("task_5", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

seeds <- 1:3


# 1. Data --------------------------------------------------------------------

data  <- load_handwriting()
split <- assign_session_split(data$meta)

train <- get_set(data, split, "train")
val   <- get_set(data, split, "val")


# 2. Configurations ----------------------------------------------------------

# Hyperparameters varied, chosen for the main problem found in Task 2 (the model
# memorises the training sessions but transfers poorly to a new session):
#   shift   - data augmentation by random shifts, mimics session-to-session drift
#   dropout - regularisation in the dense layer
#   filters - capacity of the convolutional blocks
# Learning rate and batch size stay at their Adam defaults, which are robust.
baseline <- tibble(filters = list(c(16, 32, 64)), dropout = 0.5, shift = 0)

configs <- bind_rows(
  baseline |> mutate(hyperparameter = "baseline", value = "baseline"),
  tibble(hyperparameter = "shift", shift = c(0.05, 0.1, 0.15)) |>
    mutate(filters = baseline$filters, dropout = baseline$dropout, value = as.character(shift)),
  tibble(hyperparameter = "dropout", dropout = c(0.25, 0.7)) |>
    mutate(filters = baseline$filters, shift = baseline$shift, value = as.character(dropout)),
  tibble(hyperparameter = "filters", filters = list(c(8, 16, 32), c(32, 64, 128))) |>
    mutate(dropout = baseline$dropout, shift = baseline$shift,
           value = map_chr(filters, \(f) paste(f, collapse = "-")))
) |>
  mutate(config = row_number(), .before = 1)

configs


# 3. One-at-a-time runs ------------------------------------------------------

# Roughly 30-60 seconds per run on a laptop CPU
run_configs <- function(configs, seeds) {
  configs |>
    expand_grid(seed = seeds) |>
    mutate(result = pmap(list(filters, dropout, shift, seed), \(filters, dropout, shift, seed) {
      message("filters ", paste(filters, collapse = "-"), " | dropout ", dropout,
              " | shift ", shift, " | seed ", seed)
      evaluate_config(train, val, seed, filters, dropout, shift)
    })) |>
    select(config, hyperparameter, value, seed, result) |>
    unnest(result)
}

results <- run_configs(configs, seeds)
overall <- overall_by_run(results)

summary_table <- overall |>
  summarise(
    mean_accuracy = mean(accuracy),
    sd_accuracy   = sd(accuracy),
    mean_top3     = mean(top3_accuracy),
    median_epoch  = median(best_epoch),
    .by = c(config, hyperparameter, value)
  ) |>
  mutate(gain_vs_baseline = mean_accuracy - mean_accuracy[hyperparameter == "baseline"])

summary_table |> arrange(desc(mean_accuracy))


# 4. Combined configuration --------------------------------------------------

# Best value of each hyperparameter, kept only if it beats the baseline by more
# than the seed-to-seed spread; otherwise the simpler baseline value is kept
noise <- mean(summary_table$sd_accuracy)

best_values <- summary_table |>
  filter(hyperparameter != "baseline") |>
  slice_max(mean_accuracy, n = 1, by = hyperparameter, with_ties = FALSE) |>
  filter(gain_vs_baseline > noise) |>
  left_join(configs |> select(config, filters, dropout, shift), by = "config")

combined <- baseline
for (h in best_values$hyperparameter) combined[[h]] <- best_values[[h]][best_values$hyperparameter == h]

combined <- combined |>
  mutate(config = max(configs$config) + 1, hyperparameter = "combined",
         value = paste0(paste(filters[[1]], collapse = "-"), " | ", dropout, " | ", shift))

# Only worth a run if more than one hyperparameter improved on its own
configs_all <- configs
if (nrow(best_values) > 1) {
  results     <- bind_rows(results, run_configs(combined, seeds))
  overall     <- overall_by_run(results)
  configs_all <- bind_rows(configs, combined)
}


# 5. Network depth on top of the best configuration so far -------------------

# Second stage: the number of convolutional blocks is varied around the best
# configuration from stage one. More than 4 blocks is impossible at 28 x 28
# (pooling halves the resolution 28 -> 14 -> 7 -> 3 -> 1).
stage_one_best <- overall |>
  summarise(mean_accuracy = mean(accuracy), .by = config) |>
  slice_max(mean_accuracy, n = 1, with_ties = FALSE) |>
  left_join(configs_all, by = "config")

depth_configs <- tibble(filters = list(c(16, 32), c(16, 32, 64, 128))) |>
  mutate(
    dropout        = stage_one_best$dropout,
    shift          = stage_one_best$shift,
    hyperparameter = "depth",
    value          = map_chr(filters, \(f) paste0(length(f), " blocks (", paste(f, collapse = "-"), ")")),
    config         = max(configs_all$config) + row_number()
  )

results     <- bind_rows(results, run_configs(depth_configs, seeds))
overall     <- overall_by_run(results)
configs_all <- bind_rows(configs_all, depth_configs)

# Reference for each panel of the sensitivity plot: stage one varies around the
# baseline, the depth stage around the best stage-one configuration
reference <- overall |>
  summarise(mean = mean(accuracy), .by = config) |>
  filter(config %in% c(1, stage_one_best$config)) |>
  mutate(stage = if_else(config == 1, "baseline", "stage_one_best"))

summary_table <- overall |>
  summarise(
    mean_accuracy = mean(accuracy),
    sd_accuracy   = sd(accuracy),
    mean_top3     = mean(top3_accuracy),
    median_epoch  = median(best_epoch),
    .by = c(config, hyperparameter, value)
  ) |>
  mutate(gain_vs_reference = mean_accuracy - if_else(
    hyperparameter == "depth",
    reference$mean[reference$stage == "stage_one_best"],
    reference$mean[reference$stage == "baseline"]
  ))

summary_table |> arrange(desc(mean_accuracy))


# 6. Sensitivity of the evaluation measures ----------------------------------

overall_summary <- overall |>
  summarise(mean = mean(accuracy), sd = sd(accuracy), .by = c(config, hyperparameter, value)) |>
  mutate(value = factor(value, levels = unique(value)))

reference_lines <- tibble(hyperparameter = setdiff(unique(overall_summary$hyperparameter), "baseline")) |>
  mutate(reference = if_else(
    hyperparameter == "depth",
    reference$mean[reference$stage == "stage_one_best"],
    reference$mean[reference$stage == "baseline"]
  ))

sensitivity_plot <- overall |>
  filter(hyperparameter != "baseline") |>
  mutate(value = factor(value, levels = unique(value))) |>
  ggplot(aes(value, accuracy)) +
  geom_hline(data = reference_lines, aes(yintercept = reference), linetype = "dashed", colour = "grey50") +
  geom_point(colour = "#86b6ef", size = 2, position = position_nudge(x = -0.08)) +
  geom_pointrange(
    data = overall_summary |> filter(hyperparameter != "baseline"),
    aes(y = mean, ymin = mean - sd, ymax = mean + sd),
    colour = "#2a78d6", position = position_nudge(x = 0.08)
  ) +
  facet_wrap(vars(hyperparameter), scales = "free_x") +
  scale_y_continuous(labels = scales::label_percent()) +
  labs(title = "Sensitivity of validation accuracy to the hyperparameters",
       subtitle = "Light points: single seeds. Dark: mean +- sd. Dashed: reference mean (baseline; depth: best stage-one config). Session 4",
       x = NULL, y = "Validation accuracy")

save_figure(sensitivity_plot, "sensitivity_overall", figure_dir, width = 12, height = 4.5)

# Does the effect differ by digit?
digit_summary <- results |>
  summarise(accuracy = mean(accuracy), .by = c(config, hyperparameter, value, digit)) |>
  mutate(label = if_else(hyperparameter %in% c("baseline", "combined"),
                         hyperparameter, paste(hyperparameter, value, sep = " = ")),
         label = factor(label, levels = unique(label)))

digit_plot <- ggplot(digit_summary, aes(factor(digit), label, fill = accuracy)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = scales::percent(accuracy, accuracy = 1)), size = 3) +
  scale_fill_gradient(low = "#f4f7fb", high = "#2a78d6", labels = scales::label_percent()) +
  scale_y_discrete(limits = rev) +
  labs(title = "Validation accuracy by digit and configuration",
       subtitle = "Mean over seeds, session 4", x = "Digit", y = NULL, fill = NULL)

save_figure(digit_plot, "sensitivity_by_digit", figure_dir, width = 7, height = 5.5)


# 7. Choose the final configuration ------------------------------------------

# Highest mean validation accuracy; its median best epoch fixes the number of
# epochs for the final model trained on sessions 1-4 (no early stopping possible)
final_config <- overall |>
  summarise(mean_accuracy = mean(accuracy), median_epoch = median(best_epoch), .by = config) |>
  slice_max(mean_accuracy, n = 1, with_ties = FALSE) |>
  left_join(configs_all, by = "config")

final_config

saveRDS(
  list(results = results, overall = overall, summary = summary_table,
       configs = configs_all, final_config = final_config),
  file.path(result_dir, "tuning_results.rds")
)
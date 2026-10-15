# Task 2 (diagnostic): learning curves at two learning rates
# With the default learning rate the minimum of the validation loss falls inside
# the first epoch, so only the overfitting phase is visible. A ten times smaller
# learning rate stretches the start of training over several epochs and should
# reveal the classic pattern (validation loss falls, then rises).
# Baseline architecture, no early stopping, so the full curves are recorded.

here::i_am("task_2/learning_curves.R")
source(here::here("R", "setup.R"))

use_backend("tensorflow")

figure_dir <- here::here("task_2", "cnn_plots")
result_dir <- here::here("task_2", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

seed   <- 5073
epochs <- 30


# 1. Data --------------------------------------------------------------------

data  <- load_handwriting()
split <- assign_session_split(data$meta)

train <- get_set(data, split, "train")
val   <- get_set(data, split, "val")


# 2. Train at two learning rates --------------------------------------------

learning_rates <- c(1e-3, 1e-4)

curves <- map(learning_rates, \(lr) {
  message("learning rate ", lr)
  clear_session()
  set_random_seed(seed)
  model   <- build_cnn(learning_rate = lr)
  history <- fit_cnn(model, train, val, epochs = epochs, patience = epochs, verbose = 0)

  as.data.frame(history) |>
    filter(metric %in% c("loss", "accuracy")) |>
    mutate(learning_rate = paste("Learning rate", format(lr, scientific = FALSE)))
}) |>
  list_rbind() |>
  mutate(learning_rate = factor(learning_rate, levels = unique(learning_rate))) # default rate first

# Epoch with the lowest validation loss and the highest validation accuracy
curves |>
  filter(data == "validation") |>
  summarise(
    best_epoch = if_else(first(metric) == "loss", epoch[which.min(value)], epoch[which.max(value)]),
    best_value = if_else(first(metric) == "loss", min(value), max(value)),
    .by = c(learning_rate, metric)
  )


# 3. Plot --------------------------------------------------------------------

minimum_val_loss <- curves |>
  filter(data == "validation", metric == "loss") |>
  slice_min(value, n = 1, by = learning_rate)

curve_plot <- ggplot(curves, aes(epoch, value, colour = data)) +
  geom_vline(data = minimum_val_loss |> select(learning_rate, epoch),
             aes(xintercept = epoch), linetype = "dashed", colour = "grey50") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2) +
  facet_grid(vars(metric), vars(learning_rate), scales = "free_y") +
  scale_colour_manual(values = c(training = "#2a78d6", validation = "#eb6834"),
                      labels = c(training = "Training (sessions 1-3)", validation = "Validation (session 4)")) +
  labs(title = "Learning curves at two learning rates",
       subtitle = "Baseline CNN without early stopping. Dashed line: epoch with the lowest validation loss",
       x = "Epoch", y = NULL, colour = NULL) +
  theme(legend.position = "top")

save_figure(curve_plot, "learning_curves_by_learning_rate", figure_dir, width = 10, height = 6)

saveRDS(curves, file.path(result_dir, "learning_curves.rds"))

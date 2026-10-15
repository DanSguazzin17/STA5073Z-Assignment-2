# Task 2: final CNN
# Config from task_5/cnn_tuning.R (run that first), trained on sessions 1-4 and
# tested on session 5. Also trained on sessions 1-3 to see what one extra session adds.

here::i_am("task_2/cnn_final.R")
source(here::here("R", "setup.R"))

use_backend("tensorflow")

figure_dir <- here::here("task_2", "final_plots")
result_dir <- here::here("task_2", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

seeds <- 1:5


# 1. Final configuration -----------------------------------------------------

# 4 blocks scored highest in tuning, but only because of one spike at epoch 3
# (65.8%, then 57.5%). We use 3 blocks with shift 0.1 instead (stable curve).
tuning <- readRDS(here::here("task_5", "results", "tuning_results.rds"))

final_config <- tuning$summary %>%
  filter(hyperparameter == "shift", value == "0.1") %>%
  left_join(tuning$configs %>% select(config, filters, dropout, shift), by = "config")

final_config

filters <- final_config$filters[[1]]
dropout <- final_config$dropout
shift   <- final_config$shift
epochs  <- final_config$median_epoch


# 2. Data --------------------------------------------------------------------

data  <- load_handwriting()
split <- assign_session_split(data$meta)

train_sessions_1_4 <- split %>%
  mutate(set = if_else(set == "val", "train", set)) %>%
  get_set(data = data, set_name = "train")

train_sessions_1_3 <- get_set(data, split, "train")
test               <- get_set(data, split, "test")


# 3. Train and test ----------------------------------------------------------

train_and_predict <- function(train, seed) {
  clear_session()
  set_random_seed(seed)
  model <- build_cnn(filters = filters, dropout = dropout, augment = shift > 0, shift = shift)
  fit_cnn_final(model, train, epochs = epochs, verbose = 0)
  predict_writer(model, test)
}

runs <- expand_grid(training = c("sessions 1-4", "sessions 1-3"), seed = seeds) %>%
  mutate(predictions = map2(training, seed, \(training, seed) {
    message(training, " | seed ", seed)
    train <- if (training == "sessions 1-4") train_sessions_1_4 else train_sessions_1_3
    train_and_predict(train, seed)
  }))

final_runs <- runs %>% filter(training == "sessions 1-4")


# 4. Overall and by digit ----------------------------------------------------

accuracy_by_run <- runs %>%
  mutate(by_digit = map(predictions, \(p) accuracy_by(p, digit))) %>%
  select(training, seed, by_digit) %>%
  unnest(by_digit)

# chance level is 5%
overall <- accuracy_by_run %>%
  summarise(accuracy = weighted.mean(accuracy, n), top3_accuracy = weighted.mean(top3_accuracy, n),
            .by = c(training, seed)) %>%
  summarise(across(c(accuracy, top3_accuracy), list(mean = mean, sd = sd)), .by = training)

overall

by_digit <- accuracy_by_run %>%
  summarise(across(c(accuracy, top3_accuracy), list(mean = mean, sd = sd)), .by = c(training, digit))

by_digit

digit_plot <- ggplot(by_digit, aes(factor(digit), accuracy_mean, fill = training)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_errorbar(aes(ymin = accuracy_mean - accuracy_sd, ymax = accuracy_mean + accuracy_sd),
                position = position_dodge(width = 0.75), width = 0.2, colour = "grey30") +
  scale_fill_manual(values = c("sessions 1-4" = "#2a78d6", "sessions 1-3" = "#86b6ef")) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1), expand = expansion(mult = c(0, 0.02))) +
  labs(title = "Test accuracy by digit (session 5)",
       subtitle = paste0("Mean +- sd over ", length(seeds), " seeds. Chance level: 5%"),
       x = "Digit", y = "Accuracy", fill = "Trained on")

save_figure(digit_plot, "final_accuracy_by_digit", figure_dir, width = 7, height = 4)


# 5. By writer and confusion matrix (final model, pooled over seeds) --------

final_predictions <- final_runs %>%
  select(seed, predictions) %>%
  unnest(predictions)

by_writer <- accuracy_by(final_predictions, person_id) %>% arrange(accuracy)
by_writer

writer_plot <- by_writer %>%
  mutate(writer = factor(person_id, levels = person_id)) %>% # best writer on top
  ggplot(aes(y = writer)) +
  geom_col(aes(x = accuracy), fill = "#2a78d6", width = 0.7) +
  geom_point(aes(x = top3_accuracy), shape = 21, fill = "white", colour = "grey30", size = 2) +
  geom_text(aes(x = if_else(accuracy > 0.15, accuracy - 0.01, accuracy + 0.03),
                label = scales::percent(accuracy, accuracy = 1),
                hjust = if_else(accuracy > 0.15, 1, 0),
                colour = if_else(accuracy > 0.15, "inside", "outside")), size = 3) +
  scale_colour_manual(values = c(inside = "white", outside = "grey30"), guide = "none") +
  scale_x_continuous(labels = scales::label_percent(), limits = c(0, 1.02), breaks = seq(0, 1, 0.25),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "Test accuracy by writer (session 5, final model)",
       subtitle = paste("Bars: accuracy. Points: top-3 accuracy. Pooled over", length(seeds), "seeds"),
       x = "Accuracy", y = "Writer") +
  theme(panel.grid.major.y = element_blank())

save_figure(writer_plot, "final_accuracy_by_writer", figure_dir, width = 7, height = 6)

confusion <- confusion_table(final_predictions)

confusion_plot <- ggplot(confusion, aes(factor(predicted), factor(person_id), fill = share)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  scale_fill_gradient(low = "#f4f7fb", high = "#184f95", labels = scales::label_percent()) +
  scale_y_discrete(limits = rev) +
  coord_fixed() +
  labs(title = "Confusion matrix (session 5, final model)",
       subtitle = paste("Share of each writer's images assigned to each predicted writer, pooled over", length(seeds), "seeds"),
       x = "Predicted writer", y = "True writer", fill = NULL)

save_figure(confusion_plot, "final_confusion_matrix", figure_dir, width = 7, height = 7)


# 6. Several images per writer ----------------------------------------------

by_n_images <- final_runs %>%
  mutate(aggregated = map2(predictions, seed, \(p, s) aggregated_accuracy(p, seed = s))) %>%
  select(seed, aggregated) %>%
  unnest(aggregated) %>%
  summarise(accuracy_mean = mean(accuracy), accuracy_sd = sd(accuracy), .by = k)

by_n_images

aggregation_plot <- ggplot(by_n_images, aes(k, accuracy_mean)) +
  geom_line(colour = "#2a78d6", linewidth = 0.8) +
  geom_pointrange(aes(ymin = accuracy_mean - accuracy_sd, ymax = accuracy_mean + accuracy_sd),
                  colour = "#2a78d6") +
  scale_x_continuous(breaks = by_n_images$k) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1)) +
  labs(title = "Writer identification from several images (session 5, final model)",
       subtitle = "Random images of one writer, log-probabilities summed. Mean +- sd over seeds",
       x = "Number of images", y = "Accuracy")

save_figure(aggregation_plot, "final_accuracy_by_n_images", figure_dir, width = 6, height = 4)


# 7. Save results for the report ---------------------------------------------

saveRDS(
  list(config = final_config, overall = overall, by_digit = by_digit, by_writer = by_writer,
       confusion = confusion, by_n_images = by_n_images, accuracy_by_run = accuracy_by_run,
       predictions = final_predictions),
  file.path(result_dir, "cnn_final_results.rds")
)

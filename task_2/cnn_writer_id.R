# Task 2: CNN for writer identification (baseline)
# Session split: train 1-3, validate 4, test 5. Random image split for comparison.
# Setup once per machine: renv::restore() and keras3::install_keras()

here::i_am("task_2/cnn_writer_id.R")
source(here::here("R", "setup.R"))

use_backend("tensorflow")

figure_dir <- here::here("task_2", "cnn_plots")
result_dir <- here::here("task_2", "results")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

seed <- 5073


# 1. Data and split ----------------------------------------------------------

data  <- load_handwriting()
split <- assign_session_split(data$meta)

split %>% count(set, session)
split %>% count(set, person_id) %>% count(set, n, name = "writers") # all writers in every set

train <- get_set(data, split, "train")
val   <- get_set(data, split, "val")
test  <- get_set(data, split, "test")


# 2. Model and training ------------------------------------------------------

set_random_seed(seed)
model <- build_cnn()
model

history <- fit_cnn(model, train, val)

history_plot <- as.data.frame(history) %>%
  filter(metric %in% c("loss", "accuracy"), !is.na(value)) %>% # NA after early stopping
  ggplot(aes(epoch, value, colour = data)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.5) +
  facet_wrap(vars(metric), scales = "free_y") +
  scale_colour_manual(values = c(training = "#2a78d6", validation = "#eb6834"),
                      labels = c(training = "Training (sessions 1-3)", validation = "Validation (session 4)")) +
  labs(title = "Training history", x = "Epoch", y = NULL, colour = NULL) +
  theme(legend.position = "top")

save_figure(history_plot, "training_history", figure_dir, width = 9, height = 4)


# 3. Evaluation on the test session -----------------------------------------

predictions <- predict_writer(model, test)

# chance level is 5%
overall <- accuracy_by(predictions)
overall

# by digit
by_digit <- accuracy_by(predictions, digit)
by_digit

digit_plot <- ggplot(by_digit, aes(factor(digit), accuracy)) +
  geom_col(width = 0.6, fill = "#2a78d6") +
  geom_hline(yintercept = overall$accuracy, linetype = "dashed", colour = "grey40") +
  geom_text(aes(label = scales::percent(accuracy, accuracy = 0.1)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Test accuracy by digit (session 5)",
       subtitle = "Dashed line: overall accuracy. Chance level: 5%",
       x = "Digit", y = "Accuracy")

save_figure(digit_plot, "accuracy_by_digit", figure_dir, width = 6, height = 4)

# by writer
by_writer <- accuracy_by(predictions, person_id) %>% arrange(accuracy)
by_writer

confusion_plot <- confusion_table(predictions) %>%
  ggplot(aes(factor(predicted), factor(person_id), fill = share)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  scale_fill_gradient(low = "#f4f7fb", high = "#184f95", labels = scales::label_percent()) +
  scale_y_discrete(limits = rev) +
  coord_fixed() +
  labs(title = "Confusion matrix (test session)",
       subtitle = "Share of each writer's images assigned to each predicted writer",
       x = "Predicted writer", y = "True writer", fill = NULL)

save_figure(confusion_plot, "confusion_matrix", figure_dir, width = 7, height = 7)

# combining several images of the same writer
by_n_images <- aggregated_accuracy(predictions)
by_n_images

aggregation_plot <- ggplot(by_n_images, aes(k, accuracy)) +
  geom_line(colour = "#2a78d6", linewidth = 0.8) +
  geom_point(colour = "#2a78d6", size = 2.5) +
  scale_x_continuous(breaks = by_n_images$k) +
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1)) +
  labs(title = "Writer identification from several images",
       subtitle = "Random images of one writer from the test session, log-probabilities summed",
       x = "Number of images", y = "Accuracy")

save_figure(aggregation_plot, "accuracy_by_n_images", figure_dir, width = 6, height = 4)


# 4. Comparison: random image-level split -----------------------------------

# same model, only the split differs
random_split <- assign_random_split(data$meta, seed = seed)

set_random_seed(seed)
model_random <- build_cnn()
history_random <- fit_cnn(model_random,
                          get_set(data, random_split, "train"),
                          get_set(data, random_split, "val"))

predictions_random <- predict_writer(model_random, get_set(data, random_split, "test"))

split_comparison <- bind_rows(
  session = accuracy_by(predictions, digit),
  random  = accuracy_by(predictions_random, digit),
  .id = "split"
) %>%
  select(split, digit, accuracy) %>%
  pivot_wider(names_from = split, values_from = accuracy) %>%
  mutate(gap = random - session)

split_comparison


# 5. Save results for the report ---------------------------------------------

save_model(model, file.path(result_dir, "cnn_session_split.keras"), overwrite = TRUE)

saveRDS(
  list(
    history            = as.data.frame(history),
    overall            = overall,
    by_digit           = by_digit,
    by_writer          = by_writer,
    by_n_images        = by_n_images,
    confusion          = confusion_table(predictions),
    split_comparison   = split_comparison,
    predictions        = predictions
  ),
  file.path(result_dir, "cnn_results.rds")
)

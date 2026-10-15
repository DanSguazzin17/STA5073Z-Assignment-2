# Hyperparameter tuning helpers (Task 5)

# Train one configuration with one seed, return validation accuracy by digit
# and the best epoch
evaluate_config <- function(train, val, seed, filters, dropout, shift, epochs = 50) {
  clear_session() # free memory from earlier models
  set_random_seed(seed)

  model <- build_cnn(filters = filters, dropout = dropout, augment = shift > 0, shift = shift)
  history <- fit_cnn(model, train, val, epochs = epochs, verbose = 0)

  predict_writer(model, val) %>%
    accuracy_by(digit) %>%
    mutate(
      best_epoch = which.max(history$metrics$val_accuracy),
      epochs_run = length(history$metrics$loss)
    )
}

# Overall accuracy per run (all digits have the same number of images)
overall_by_run <- function(results) {
  results %>%
    summarise(
      accuracy      = weighted.mean(accuracy, n),
      top3_accuracy = weighted.mean(top3_accuracy, n),
      best_epoch    = first(best_epoch),
      .by = c(config, hyperparameter, value, seed)
    )
}

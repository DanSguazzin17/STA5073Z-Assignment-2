# Evaluation helpers for writer identification

# One row per image with true and predicted writer; class probabilities as attribute
predict_writer <- function(model, set) {
  probs <- predict(model, set$x, verbose = 0)

  set$meta |>
    mutate(
      predicted = max.col(probs, ties.method = "first"),
      prob_true = probs[cbind(seq_len(nrow(probs)), person_id)],
      in_top3   = rowSums(probs > prob_true) < 3,
      correct   = predicted == person_id
    ) |>
    structure(probs = probs)
}

accuracy_by <- function(predictions, ...) {
  predictions |>
    group_by(...) |>
    summarise(
      n             = n(),
      accuracy      = mean(correct),
      top3_accuracy = mean(in_top3),
      .groups       = "drop"
    )
}

# Share of each true writer's images assigned to each predicted writer
confusion_table <- function(predictions, n_classes = 20) {
  predictions |>
    count(person_id, predicted) |>
    complete(person_id = seq_len(n_classes), predicted = seq_len(n_classes), fill = list(n = 0)) |>
    group_by(person_id) |>
    mutate(share = n / sum(n)) |>
    ungroup()
}

# Writer identification from several images: draw k test images of the same writer,
# add their log-probabilities and predict the writer with the highest sum. This
# mirrors identifying a person from a document rather than from a single digit.
aggregated_accuracy <- function(predictions, k_values = c(1, 2, 5, 10, 25), n_draws = 200, seed = 5073) {
  log_probs <- log(pmax(attr(predictions, "probs"), 1e-12))
  rows_by_writer <- split(seq_len(nrow(predictions)), predictions$person_id)

  withr::with_seed(seed, {
    map(k_values, \(k) {
      hits <- imap(rows_by_writer, \(rows, writer) {
        replicate(n_draws, {
          drawn <- sample(rows, k)
          which.max(colSums(log_probs[drawn, , drop = FALSE])) == as.integer(writer)
        })
      })
      tibble(k = k, accuracy = mean(unlist(hits)))
    }) |>
      list_rbind()
  })
}

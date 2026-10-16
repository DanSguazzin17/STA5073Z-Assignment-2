# ---------------------------------------------------------------------------
# Task 3: Siamese network for same-writer matching
# Task 5: limited hyperparameter search for the Siamese network
#
# Goal: given two handwriting images, predict whether the SAME person wrote
# both of them. Unlike the CNN (task_2), this doesn't classify "which of the
# 20 writers" -- it learns a general notion of handwriting similarity, which
# in principle works even for a writer the model never trained on.
#
# This follows the worked example in:
#   "10 Object Detection and Metric Learning/Metric Learning with Siamese
#   Networks.qmd" -- read that alongside this script.
#
# Overall plan (matches the plan agreed with Lucas for both models):
#   1. Train on sessions 1-3, tune/check overfitting on session 4 (validation).
#      Session 5 (test) is NOT touched during this phase.
#   2. Try a few hyperparameter settings, compare them using ONLY validation.
#      Pick whichever setting scores best on validation.
#   3. Retrain ONE final model using the winning settings, now on sessions
#      1-4 combined (more training data, since we no longer need a separate
#      validation set -- we already used it to pick the settings).
#   4. Evaluate that final model ONCE on session 5 (test). This is the number
#      that goes in the report and gets compared against the CNN.
# ---------------------------------------------------------------------------

source(here::here("R", "setup.R"))   # loads packages + shared functions (data.R, plotting.R, ...)

set.seed(5073)
keras3::set_random_seed(5073)   # seeds TensorFlow/Keras weight init + shuffling too, so reruns are reproducible


# 1. Load data and apply the shared session split ----------------------------
# Train = sessions 1-3, validation = session 4, test = session 5.
# (assign_session_split() lives in R/data.R, written by Lucas -- see README)

data  <- load_handwriting()
split <- assign_session_split(data$meta)

split |> count(set)   # sanity check: should be ~15000 train / 5000 val / 5000 test

train_rows <- which(split$set == "train")          # sessions 1-3, used for hyperparameter search
val_rows   <- which(split$set == "val")             # session 4, used to pick the best hyperparameters
test_rows  <- which(split$set == "test")            # session 5, touched only ONCE, right at the end

final_train_rows <- which(split$session != 5)       # sessions 1-4 combined, used for the final model


# 2. Build same/different-writer pairs, controlling for digit ----------------
# The brief asks us to think about whether pair construction should control
# for digit identity. We do: a pair's two images are ALWAYS the same digit,
# so the network can't take the shortcut of learning "these look like the
# same digit" as a stand-in for "these were written by the same person."
#
# `ids` restricts which rows of the metadata we're allowed to draw from
# (e.g. only training rows), so we never mix images across train/val/test.

make_pairs <- function(ids, n, meta) {
  a <- sample(ids, n, replace = TRUE)

  b <- vapply(a, function(i) {
    same_writer <- runif(1) < 0.5                       # coin flip: same or different writer
    pool <- ids[meta$digit[ids] == meta$digit[i]]        # same digit always
    pool <- if (same_writer) {
      pool[meta$person_id[pool] == meta$person_id[i]]    # same digit AND same writer
    } else {
      pool[meta$person_id[pool] != meta$person_id[i]]    # same digit, DIFFERENT writer
    }
    sample(setdiff(pool, i), 1)                          # never pair an image with itself
  }, integer(1))

  cbind(a, b)
}

# Turn a matrix of index-pairs into the actual images + labels Keras needs.
# y = 1 means "same writer", y = 0 means "different writer".
make_batch <- function(pairs, images, meta) {
  y <- matrix(as.integer(meta$person_id[pairs[, 1]] == meta$person_id[pairs[, 2]]), ncol = 1)
  list(
    left  = images[pairs[, 1], , , , drop = FALSE],
    right = images[pairs[, 2], , , , drop = FALSE],
    y     = y
  )
}

# How many pairs: kept modest so this trains quickly on a laptop CPU,
# matching the brief's "models trained in under a minute" guidance.
train_pairs <- make_batch(make_pairs(train_rows, 8000, split), data$x, split)
val_pairs   <- make_batch(make_pairs(val_rows,   2000, split), data$x, split)
test_pairs  <- make_batch(make_pairs(test_rows,  2000, split), data$x, split)

mean(train_pairs$y)   # sanity check: should be close to 0.5 (balanced same/different)


# 3. Build the Siamese model as a FUNCTION ------------------------------------
# Same idea as Lucas's build_cnn(): every hyperparameter we might want to
# vary (embedding size, margin) is a function argument with a sensible
# default, instead of being hard-coded. That's what makes step 5 (the
# hyperparameter search) possible without rewriting the model each time.

build_siamese <- function(embedding_dim = 16,
                           margin        = 1,
                           filters       = c(16, 32),
                           learning_rate = 1e-3) {

  # The shared encoder: one small CNN, used on BOTH images with the SAME
  # weights. It turns a 28x28 image into `embedding_dim` numbers summarising
  # its handwriting style.
  image_in <- layer_input(shape = c(28, 28, 1))
  embedding <- image_in |>
    layer_conv_2d(filters = filters[1], kernel_size = 3, activation = "relu") |>
    layer_max_pooling_2d(pool_size = 2) |>
    layer_conv_2d(filters = filters[2], kernel_size = 3, activation = "relu") |>
    layer_global_average_pooling_2d() |>
    layer_dense(units = embedding_dim)
  encoder <- keras_model(image_in, embedding)

  # Full Siamese model: two images in, one distance out. encoder(a) and
  # encoder(b) reuse the SAME encoder weights for both images. Small
  # distance = network thinks "same person."
  input_a <- layer_input(shape = c(28, 28, 1))
  input_b <- layer_input(shape = c(28, 28, 1))
  distance <- list(encoder(input_a), encoder(input_b)) |>
    layer_lambda(f = function(z) {
      op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2, keepdims = TRUE) + 1e-7)
    })
  model <- keras_model(list(input_a, input_b), distance)

  # Contrastive loss, with the margin passed in as an argument.
  # y = 1 (same writer): loss = distance^2 -> punished for being far apart.
  # y = 0 (different writer): loss = 0 once distance passes the margin --
  # different writers just need to be "far enough" apart, not infinitely far.
  contrastive_loss <- function(y, d) {
    op_mean(y * op_square(d) + (1 - y) * op_square(op_maximum(margin - d, 0)))
  }

  model |> compile(
    optimizer = optimizer_adam(learning_rate),
    loss      = contrastive_loss
  )

  model
}


# 4. Helper: pick the best distance threshold using VALIDATION pairs ---------
# The model only outputs a distance, not a yes/no answer. We try a range of
# candidate thresholds and keep whichever gives the best "balanced accuracy"
# -- averaging how often we correctly spot same-writer pairs and how often
# we correctly spot different-writer pairs, so a model can't win just by
# always guessing "different."

best_threshold <- function(distance, same) {
  candidates <- sort(unique(distance))
  scores <- vapply(candidates, function(cut) {
    predicted_same <- distance < cut
    same_rate      <- mean(predicted_same[same])
    different_rate <- mean(!predicted_same[!same])
    (same_rate + different_rate) / 2
  }, numeric(1))
  list(threshold = candidates[which.max(scores)], balanced_accuracy = max(scores))
}


# 5. Train ONE candidate model and score it on validation ---------------------
# Used by the hyperparameter search below: build a model with the given
# settings, train it on sessions 1-3, then score it on session 4 only.
# Test pairs are never touched here.

train_one_config <- function(embedding_dim, margin, epochs = 10, batch_size = 64) {
  model <- build_siamese(embedding_dim = embedding_dim, margin = margin)

  model |> fit(
    list(train_pairs$left, train_pairs$right), train_pairs$y,
    validation_data = list(list(val_pairs$left, val_pairs$right), val_pairs$y),
    epochs     = epochs,
    batch_size = batch_size,
    verbose    = 0
  )

  val_distance <- as.numeric(predict(model, list(val_pairs$left, val_pairs$right), verbose = 0))
  val_same     <- as.logical(val_pairs$y[, 1])
  best         <- best_threshold(val_distance, val_same)

  tibble(
    embedding_dim     = embedding_dim,
    margin            = margin,
    threshold         = best$threshold,
    val_balanced_accuracy = best$balanced_accuracy
  )
}


# 6. The hyperparameter search itself (Task 5) --------------------------------
# A "limited" search, as the brief asks for: we vary two hyperparameters
# one at a time around our original (embedding_dim = 16, margin = 1)
# baseline, so we can see how sensitive performance is to each one
# individually, without training dozens of models.

candidate_configs <- tribble(
  ~label,              ~embedding_dim, ~margin,
  "baseline",           16,             1,
  "smaller embedding",  8,              1,
  "larger embedding",   32,             1,
  "smaller margin",     16,             0.5,
  "larger margin",      16,             2
)

search_results <- candidate_configs |>
  select(-label) |>
  pmap(train_one_config) |>
  bind_rows() |>
  mutate(label = candidate_configs$label, .before = 1)

search_results   # one row per candidate -- compare val_balanced_accuracy across them

# A quick plot showing how sensitive results are to each setting (for the report)
sensitivity_plot <- search_results |>
  ggplot(aes(x = label, y = val_balanced_accuracy)) +
  geom_col() +
  labs(
    title = "Siamese network: hyperparameter sensitivity (validation set)",
    x = NULL, y = "Validation balanced accuracy"
  ) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

save_figure(sensitivity_plot, "hyperparameter_sensitivity", here::here("task_3", "results"),
            width = 7, height = 4.5)

# Pick the winning configuration -- best validation balanced accuracy.
best_config <- search_results |> slice_max(val_balanced_accuracy, n = 1, with_ties = FALSE)
best_config


# 7. Retrain ONE final model on sessions 1-4 combined --------------------------
# We've now picked our settings using validation, so there's no more reason
# to hold session 4 back -- we fold it back into training for the final
# model. The threshold stays the one we locked in during the search above
# (chosen using validation data only, never test).

final_pairs <- make_batch(make_pairs(final_train_rows, 10000, split), data$x, split)

final_model <- build_siamese(
  embedding_dim = best_config$embedding_dim,
  margin        = best_config$margin
)

final_history <- final_model |> fit(
  list(final_pairs$left, final_pairs$right), final_pairs$y,
  epochs     = 10,
  batch_size = 64
)

plot(final_history)


# 8. Evaluate the final model ONCE on the TEST pairs ---------------------------
# Important: we do NOT re-pick the threshold here -- that would be cheating
# (using test information to make a decision). We only USE the threshold
# chosen back in step 6.

threshold <- best_config$threshold

test_distance  <- as.numeric(predict(final_model, list(test_pairs$left, test_pairs$right)))
test_same      <- as.logical(test_pairs$y[, 1])
predicted_same <- test_distance < threshold

test_same_rate      <- mean(predicted_same[test_same])       # correctly caught same-writer pairs
test_different_rate <- mean(!predicted_same[!test_same])     # correctly caught different-writer pairs
test_balanced_accuracy <- (test_same_rate + test_different_rate) / 2

test_balanced_accuracy
test_same_rate
test_different_rate


# 9. Save a results plot --------------------------------------------------------
# Distance distributions for same vs. different pairs, with the chosen
# threshold marked -- a good "does this even make sense" sanity check, and
# useful evidence for the report.

distance_plot <- tibble(
  distance = test_distance,
  pair     = ifelse(test_same, "Same writer", "Different writer")
) |>
  ggplot(aes(distance, fill = pair)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  geom_vline(xintercept = threshold, linetype = "dashed") +
  labs(
    title    = "Siamese network: distance between embeddings on test pairs",
    subtitle = paste0("Dashed line = chosen threshold (", round(threshold, 3), ")"),
    x = "Distance", y = "Number of pairs", fill = NULL
  )

save_figure(distance_plot, "test_distance_distribution", here::here("task_3", "results"),
            width = 8, height = 4.5)


# 10. Extra check: compare train vs. validation vs. test balanced accuracy -----
# This is the direct evidence for whether the model is overfitting.
# Overfitting would show up as train accuracy being MUCH higher than
# validation/test accuracy (the model "memorised" training pairs instead of
# learning general handwriting style). If all three numbers are similar,
# that confirms the high accuracy is real, not memorisation.

train_distance <- as.numeric(predict(final_model, list(final_pairs$left, final_pairs$right)))
train_same     <- as.logical(final_pairs$y[, 1])
predicted_same_train <- train_distance < threshold   # same locked-in threshold, not re-picked

train_balanced_accuracy <- (mean(predicted_same_train[train_same]) +
                               mean(!predicted_same_train[!train_same])) / 2

accuracy_comparison <- tibble(
  set                = c("Train (sessions 1-4)", "Validation (session 4, during search)", "Test (session 5)"),
  balanced_accuracy  = c(train_balanced_accuracy, best_config$val_balanced_accuracy, test_balanced_accuracy)
)

accuracy_comparison   # all three should be close together -- that's the "not overfitting" evidence


# 11. Does the extra session of training data actually help? ------------------
# Same idea as what Lucas did for the CNN: train a version using ONLY sessions
# 1-3 (same as the original hyperparameter search), and compare its test
# performance against the final model (trained on sessions 1-4). This shows
# how much that one extra recording session of training data is worth --
# useful evidence for Task 6 ("how session changes affect performance").

sessions_1_3_model <- build_siamese(
  embedding_dim = best_config$embedding_dim,
  margin        = best_config$margin
)

sessions_1_3_model |> fit(
  list(train_pairs$left, train_pairs$right), train_pairs$y,   # train_pairs = sessions 1-3 only
  epochs     = 10,
  batch_size = 64
)

sessions_1_3_distance <- as.numeric(predict(sessions_1_3_model, list(test_pairs$left, test_pairs$right)))
predicted_same_1_3    <- sessions_1_3_distance < threshold    # same locked-in threshold throughout

sessions_1_3_balanced_accuracy <- (mean(predicted_same_1_3[test_same]) +
                                      mean(!predicted_same_1_3[!test_same])) / 2

training_data_comparison <- tibble(
  training_data      = c("Sessions 1-3 only", "Sessions 1-4 (final model)"),
  test_balanced_accuracy = c(sessions_1_3_balanced_accuracy, test_balanced_accuracy)
)

training_data_comparison   # a bigger gap here means the extra session of data mattered more


# 12. Quick exploration: forced cross-session matching (sessions 4 vs. 5) -------
# Everything above is unchanged. This is a small, additional check: how well
# does the model match handwriting across two sessions it has genuinely
# never seen, rather than relying on the (easier) within-session pairs that
# make up the normal test set? Same-writer pairs are forced to have one
# image from session 4 and one from session 5. Evaluated with
# sessions_1_3_model (trained only on sessions 1-3, so both session 4 and
# session 5 are unseen to it) and the same locked-in threshold used
# throughout. This is a quick exploratory check, not a full re-run of the
# pipeline -- kept separate from the main results above.

make_cross_session_pairs <- function(n, meta) {
  session4_ids <- which(meta$session == 4)
  session5_ids <- which(meta$session == 5)

  a <- sample(session4_ids, n, replace = TRUE)
  b <- vapply(a, function(i) {
    same_writer <- runif(1) < 0.5
    pool <- session5_ids[meta$digit[session5_ids] == meta$digit[i]]
    pool <- if (same_writer) {
      pool[meta$person_id[pool] == meta$person_id[i]]
    } else {
      pool[meta$person_id[pool] != meta$person_id[i]]
    }
    sample(pool, 1)
  }, integer(1))

  cbind(a, b)
}

cross_session_pairs <- make_batch(make_cross_session_pairs(2000, split), data$x, split)

cross_session_distance <- as.numeric(predict(sessions_1_3_model, list(cross_session_pairs$left, cross_session_pairs$right)))
cross_session_same     <- as.logical(cross_session_pairs$y[, 1])
predicted_same_cross   <- cross_session_distance < threshold

cross_session_same_rate      <- mean(predicted_same_cross[cross_session_same])
cross_session_different_rate <- mean(!predicted_same_cross[!cross_session_same])
cross_session_balanced_accuracy <- (cross_session_same_rate + cross_session_different_rate) / 2

cross_session_balanced_accuracy
cross_session_same_rate
cross_session_different_rate


# 13. Quick exploration: balanced accuracy broken down by digit -------------
# For comparison against the CNN's per-digit breakdown in @sec-cnn: does any
# digit carry more writer information for the Siamese network either?
# Builds a fresh set of same-digit test pairs for each digit separately
# (ids restricted to one digit within the test session, session 5), then
# evaluates with the already-trained final_model and the same locked-in
# threshold used throughout. This is a quick exploratory check using new
# pairs, not a breakdown of the original test_pairs -- kept separate from
# the main results above, nothing above this is affected.

digit_balanced_accuracy <- function(d, n = 400) {
  ids_d    <- test_rows[split$digit[test_rows] == d]
  pairs_d  <- make_batch(make_pairs(ids_d, n, split), data$x, split)
  dist_d   <- as.numeric(predict(final_model, list(pairs_d$left, pairs_d$right)))
  same_d   <- as.logical(pairs_d$y[, 1])
  pred_d   <- dist_d < threshold
  (mean(pred_d[same_d]) + mean(!pred_d[!same_d])) / 2
}

siamese_digit_accuracy <- sapply(sort(unique(split$digit)), digit_balanced_accuracy)
names(siamese_digit_accuracy) <- sort(unique(split$digit))
siamese_digit_accuracy

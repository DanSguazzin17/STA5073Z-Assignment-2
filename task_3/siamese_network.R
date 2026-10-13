# ---------------------------------------------------------------------------
# Task 3: Siamese network for same-writer matching
#
# Goal: given two handwriting images, predict whether the SAME person wrote
# both of them. Unlike the CNN (task_2), this doesn't classify "which of the
# 20 writers" -- it learns a general notion of handwriting similarity, which
# in principle works even for a writer the model never trained on.
#
# This follows the worked example in:
#   "10 Object Detection and Metric Learning/Metric Learning with Siamese
#   Networks.qmd" -- read that alongside this script.
# ---------------------------------------------------------------------------

source(here::here("R", "setup.R"))   # loads packages + shared functions (data.R, plotting.R, ...)

set.seed(5073)


# 1. Load data and apply the shared session split ----------------------------
# Train = sessions 1-3, validation = session 4, test = session 5.
# (assign_session_split() lives in R/data.R, written by Lucas -- see README)

data  <- load_handwriting()
split <- assign_session_split(data$meta)

split |> count(set)   # sanity check: should be ~15000 train / 5000 val / 5000 test

train_rows <- which(split$set == "train")
val_rows   <- which(split$set == "val")
test_rows  <- which(split$set == "test")


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


# 3. Shared encoder: one CNN, used on both images -----------------------------
# This is the "Siamese" part -- both images go through the SAME network
# (same weights), turning each 28x28 image into a short list of numbers
# (an "embedding") that summarises its handwriting style.

image_in <- layer_input(shape = c(28, 28, 1))

embedding <- image_in |>
  layer_conv_2d(filters = 16, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d(pool_size = 2) |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") |>
  layer_global_average_pooling_2d() |>
  layer_dense(units = 16)                      # the embedding: 16 numbers per image

encoder <- keras_model(image_in, embedding)


# 4. Full Siamese model: two images in, one distance out ---------------------
# encoder(a) and encoder(b) reuse the SAME encoder weights for both images.
# We then compute the Euclidean (straight-line) distance between the two
# embeddings -- small distance = network thinks "same person."

input_a <- layer_input(shape = c(28, 28, 1))
input_b <- layer_input(shape = c(28, 28, 1))

distance <- list(encoder(input_a), encoder(input_b)) |>
  layer_lambda(f = function(z) {
    op_sqrt(op_sum(op_square(z[[1]] - z[[2]]), axis = 2, keepdims = TRUE) + 1e-7)
  })

siamese_model <- keras_model(list(input_a, input_b), distance)


# 5. Contrastive loss ----------------------------------------------------------
# y = 1 (same writer): loss = distance^2 -> punished for being far apart.
# y = 0 (different writer): loss = 0 once distance passes the margin (1) --
# different writers just need to be "far enough" apart, not infinitely far.

contrastive_loss <- function(y, d) {
  margin <- 1
  op_mean(y * op_square(d) + (1 - y) * op_square(op_maximum(margin - d, 0)))
}

siamese_model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss      = contrastive_loss
)


# 6. Train ----------------------------------------------------------------------

history <- siamese_model |> fit(
  list(train_pairs$left, train_pairs$right), train_pairs$y,
  validation_data = list(list(val_pairs$left, val_pairs$right), val_pairs$y),
  epochs     = 10,
  batch_size = 64
)

plot(history)


# 7. Choose a distance threshold using the VALIDATION pairs --------------------
# The model only outputs a distance, not a yes/no answer. We try a range of
# candidate thresholds on the validation pairs and keep whichever gives the
# best "balanced accuracy" -- averaging how often we correctly spot same-
# writer pairs and how often we correctly spot different-writer pairs, so a
# model can't win just by always guessing "different."

val_distance <- as.numeric(predict(siamese_model, list(val_pairs$left, val_pairs$right)))
val_same     <- as.logical(val_pairs$y[, 1])

balanced_accuracy_at <- function(cut) {
  predicted_same <- val_distance < cut
  same_rate      <- mean(predicted_same[val_same])
  different_rate <- mean(!predicted_same[!val_same])
  (same_rate + different_rate) / 2
}

candidates <- sort(unique(val_distance))
scores     <- vapply(candidates, balanced_accuracy_at, numeric(1))
threshold  <- candidates[which.max(scores)]

threshold                 # the chosen cut-off distance
max(scores)                # the validation balanced accuracy at that cut-off


# 8. Evaluate on the TEST pairs, using the threshold we already locked in ------
# Important: we do NOT re-pick the threshold here -- that would be cheating
# (using test information to make a decision). We only USE the threshold.

test_distance <- as.numeric(predict(siamese_model, list(test_pairs$left, test_pairs$right)))
test_same     <- as.logical(test_pairs$y[, 1])
predicted_same <- test_distance < threshold

test_same_rate      <- mean(predicted_same[test_same])       # correctly caught same-writer pairs
test_different_rate <- mean(!predicted_same[!test_same])     # correctly caught different-writer pairs
test_balanced_accuracy <- (test_same_rate + test_different_rate) / 2

test_balanced_accuracy
test_same_rate
test_different_rate


# 9. Save a results plot -------------------------------------------------------
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

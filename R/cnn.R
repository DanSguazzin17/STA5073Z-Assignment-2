# CNN for writer identification. Hyperparameters are arguments so that Task 5
# can vary them without touching the architecture code.

build_cnn <- function(filters       = c(16, 32, 64),
                      dense_units   = 64,
                      dropout       = 0.5,
                      learning_rate = 1e-3,
                      augment       = FALSE,
                      shift         = 0.1,
                      n_classes     = 20,
                      input_shape   = c(28, 28, 1)) {
  inputs <- keras_input(shape = input_shape)
  x <- inputs
  
  # Random shifts are only applied during training, never at prediction time
  if (augment) {
    x <- x |> layer_random_translation(
      height_factor = shift, width_factor = shift, fill_mode = "constant"
    )
  }
  
  # Each block halves the resolution (28 -> 14 -> 7 -> 3) and learns more filters
  for (n_filters in filters) {
    x <- x |>
      layer_conv_2d(n_filters, kernel_size = 3, padding = "same", activation = "relu") |>
      layer_max_pooling_2d(pool_size = 2)
  }
  
  outputs <- x |>
    layer_flatten() |>
    layer_dense(dense_units, activation = "relu") |>
    layer_dropout(dropout) |>
    layer_dense(n_classes, activation = "softmax")
  
  model <- keras_model(inputs, outputs)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate),
    loss      = "sparse_categorical_crossentropy",
    metrics   = list(
      metric_sparse_categorical_accuracy(name = "accuracy"),
      metric_sparse_top_k_categorical_accuracy(k = 3, name = "top3_accuracy")
    )
  )
  
  model
}

# Early stopping on the validation session keeps the weights of the best epoch.
# Validation accuracy is monitored rather than validation loss: on a new session
# the loss rises through over-confident wrong predictions while accuracy still
# improves. Keras models are modified in place, so `model` is trained afterwards.
fit_cnn <- function(model, train, val, epochs = 50, batch_size = 64, patience = 5,
                    monitor = "val_accuracy", verbose = 2) {
  model |> fit(
    x               = train$x,
    y               = train$y,
    validation_data = list(val$x, val$y),
    epochs          = epochs,
    batch_size      = batch_size,
    callbacks       = list(callback_early_stopping(
      monitor = monitor, patience = patience, restore_best_weights = TRUE
    )),
    verbose         = verbose
  )
}

# Final model: retrain on training + validation sessions for a fixed number of
# epochs (chosen beforehand with early stopping), as no validation set is left
fit_cnn_final <- function(model, train, epochs, batch_size = 64, verbose = 2) {
  model |> fit(
    x          = train$x,
    y          = train$y,
    epochs     = epochs,
    batch_size = batch_size,
    verbose    = verbose
  )
}
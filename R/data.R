# Data loading and train/validation/test splits shared across all tasks

load_handwriting <- function(path = here::here("data", "handwriting.rds")) {
  data <- readRDS(path)
  list(
    x    = data$images / 255, # image x row x col x channel, scaled to [0, 1]
    meta = as_tibble(data$metadata)
  )
}

# Main split: whole sessions are held out, so every writer appears in every set
# but no recording session is shared between sets
assign_session_split <- function(meta, val_session = 4, test_session = 5) {
  stopifnot(val_session != test_session)

  meta |>
    mutate(set = case_when(
      session == test_session ~ "test",
      session == val_session  ~ "val",
      .default                = "train"
    ))
}

# Naive comparison split: images are assigned at random, stratified by writer and
# digit. Near-identical images from one session end up in both training and test,
# so this split is expected to be optimistic (Task 4)
assign_random_split <- function(meta, prop = c(train = 0.6, val = 0.2, test = 0.2), seed = 5073) {
  stopifnot(isTRUE(all.equal(sum(prop), 1)))
  cutoffs <- cumsum(prop)

  withr::with_seed(seed, {
    meta |>
      group_by(person_id, digit) |>
      mutate(
        quantile = sample(n()) / n(),
        set      = names(prop)[findInterval(quantile, cutoffs, left.open = TRUE) + 1]
      ) |>
      ungroup() |>
      select(-quantile)
  })
}

# Model inputs for one set: images, zero-based writer labels and metadata
get_set <- function(data, split, set_name) {
  rows <- which(split$set == set_name)
  list(
    x    = data$x[rows, , , , drop = FALSE],
    y    = split$person_id[rows] - 1L,
    meta = split[rows, ]
  )
}

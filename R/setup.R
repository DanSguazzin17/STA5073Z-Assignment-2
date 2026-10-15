# Loads packages and shared functions for all task scripts.
# Packages are installed once per machine via renv::restore() and, for the
# Python backend of keras3, keras3::install_keras() (see README).

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(keras3)
})

reticulate::py_require(python_version = "3.12")

list.files(here::here("R"), pattern = "\\.R$", full.names = TRUE) |>
  keep(\(file) basename(file) != "setup.R") |>
  walk(source)

theme_set(theme_assignment())
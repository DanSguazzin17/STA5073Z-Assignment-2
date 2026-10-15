# Loads packages and the shared functions in R/
# Install once per machine: renv::restore() and keras3::install_keras()

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(keras3)
})

reticulate::py_require(python_version = "3.12")

list.files(here::here("R"), pattern = "\\.R$", full.names = TRUE) %>%
  keep(\(file) basename(file) != "setup.R") %>%
  walk(source)

theme_set(theme_assignment())
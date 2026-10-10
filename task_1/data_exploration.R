# Task 1: Data exploration
# Explore images and metadata, show examples across people, digits and sessions,
# and look for sources of variation and limitations of the simulated data.

rm(list=objects())

library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)

# here() resolves paths from the project root (the folder holding the .Rproj file),
# so the script runs unchanged on every machine and from any working directory

# Paths are resolved from the project root, so the script runs unchanged on every machine.
# Requires the working directory to be inside the repo (open the .Rproj file).
here::i_am("task_1/data_exploration.R")

data_path  <- here::here("data", "handwriting.rds")
figure_dir <- here::here("task_1", "exploration_plots")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

image_size <- 28
session_colours <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

theme_set(theme_minimal(base_size = 11))


# 1. Load data and check structure ------------------------------------------

data   <- readRDS(data_path)
images <- data$images[, , , 1] # image x row x col, integer 0-255
meta   <- as_tibble(data$metadata)

dim(data$images)
typeof(data$images)
range(images)
anyNA(images)
glimpse(meta)

stopifnot(identical(as.integer(meta$image_id), seq_len(nrow(meta))))

# Balanced design: every writer x digit x session cell should hold 50 replicates
meta |> summarise(across(-image_id, n_distinct))
meta |> count(person_id, digit, session, name = "images_per_cell") |> count(images_per_cell)

# One row per image, pixels scaled to [0, 1]; column j is pixel (row, col) with
# j = (col - 1) * image_size + row
pixels    <- matrix(images, nrow = nrow(meta)) / 255
pixel_row <- rep(seq_len(image_size), times = image_size)
pixel_col <- rep(seq_len(image_size), each = image_size)


# Helpers --------------------------------------------------------------------

# Long format (one row per pixel) for selected images, with their metadata
images_long <- function(ids) {
  expand_grid(col = seq_len(image_size), row = seq_len(image_size), image_id = ids) |>
    mutate(value = as.vector(images[ids, , ]) / 255) |>
    left_join(meta, by = "image_id")
}

# Pixel-wise mean and sd image for every combination of the grouping variables
summarise_images <- function(...) {
  group  <- meta |> group_by(...) |> group_indices()
  keys   <- meta |> distinct(...) |> arrange(...) |> mutate(group = row_number())
  n      <- tabulate(group)
  mean   <- rowsum(pixels, group) / n
  second <- rowsum(pixels^2, group) / n
  
  expand_grid(col = seq_len(image_size), row = seq_len(image_size), group = keys$group) |>
    mutate(mean = as.vector(mean), sd = as.vector(sqrt(pmax(second - mean^2, 0)))) |>
    left_join(keys, by = "group")
}

plot_images <- function(long, rows, cols, fill = value, title = NULL, subtitle = NULL) {
  ggplot(long, aes(col, row, fill = {{ fill }})) +
    geom_raster() +
    facet_grid(vars({{ rows }}), vars({{ cols }}), labeller = label_both) +
    scale_y_reverse() +
    scale_fill_gradient(low = "white", high = "black", guide = "none") +
    coord_fixed(expand = FALSE) +
    labs(title = title, subtitle = subtitle) +
    theme_void(base_size = 9) +
    theme(
      panel.border = element_rect(colour = "grey80", fill = NA),
      panel.spacing = unit(1, "pt"),
      strip.text.y = element_text(angle = 0, hjust = 0),
      plot.title = element_text(face = "bold", margin = margin(b = 4)),
      plot.subtitle = element_text(margin = margin(b = 6))
    )
}

save_figure <- function(plot, name, width, height) {
  ggsave(file.path(figure_dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 150, bg = "white")
  if (interactive()) print(plot)
  invisible(plot)
}


# 2. Example images ----------------------------------------------------------

# Every writer x every digit (first image of session 1)
meta |>
  filter(session == 1, replicate == 1) |>
  pull(image_id) |>
  images_long() |>
  plot_images(digit, person_id, title = "One example per writer and digit (session 1)") |>
  save_figure("examples_writer_by_digit", width = 16, height = 5)

# Every writer x every session, one figure per digit
walk(sort(unique(meta$digit)), \(d) {
  meta |>
    filter(digit == d, replicate == 1) |>
    pull(image_id) |>
    images_long() |>
    plot_images(session, person_id, title = paste("Digit", d, "- one example per writer and session")) |>
    save_figure(paste0("examples_writer_by_session_digit_", d), width = 16, height = 5)
})

# Replicates within a session: how much does a single writer vary from image to image?
plot_replicates <- function(person, d, n_replicates = 12) {
  meta |>
    filter(person_id == person, digit == d, replicate <= n_replicates) |>
    pull(image_id) |>
    images_long() |>
    plot_images(session, replicate,
                title = paste0("Writer ", person, ", digit ", d, " - replicates within each session"))
}

set.seed(1234)
example_writers <- sort(sample(unique(meta$person_id), 3))

walk(example_writers, \(p) {
  plot_replicates(p, d = 2) |>
    save_figure(paste0("examples_replicates_writer_", p), width = 11, height = 5)
})


# 3. Mean and sd images ------------------------------------------------------

# Average digit and pixel-wise variability over the whole data set
digit_summary <- summarise_images(digit) |>
  pivot_longer(c(mean, sd), names_to = "statistic")

plot_images(digit_summary, statistic, digit, title = "Mean and sd image per digit") |>
  save_figure("mean_sd_by_digit", width = 8, height = 3.6)

# A writer's "style template": mean image per writer and digit, pooled over sessions
summarise_images(person_id, digit) |>
  plot_images(digit, person_id, fill = mean, title = "Mean image per writer and digit") |>
  save_figure("mean_writer_by_digit", width = 16, height = 5)

# Session shift: mean image per writer and session for a single digit
summarise_images(person_id, digit, session) |>
  filter(digit == 2) |>
  plot_images(session, person_id, fill = mean, title = "Digit 2 - mean image per writer and session") |>
  save_figure("mean_writer_by_session_digit_2", width = 16, height = 5)


# 4. Simple image features ---------------------------------------------------

ink      <- rowSums(pixels)
centre_x <- drop(pixels %*% pixel_col) / ink
centre_y <- drop(pixels %*% pixel_row) / ink
var_y    <- drop(pixels %*% pixel_row^2) / ink - centre_y^2
cov_xy   <- drop(pixels %*% (pixel_col * pixel_row)) / ink - centre_x * centre_y

features <- meta |>
  mutate(
    ink_total      = ink,
    ink_area       = rowSums(pixels > 0),
    mean_intensity = ink_total / ink_area,
    centre_x       = centre_x,
    centre_y       = centre_y,
    width          = rowSums(colSums(aperm(images, c(2, 1, 3))) > 0), # columns containing ink
    height         = rowSums(rowSums(images, dims = 2) > 0),          # rows containing ink
    slant          = -cov_xy / var_y # horizontal shift per row moved up; > 0 leans right
  )

feature_names <- setdiff(names(features), names(meta))

features_long <- features |>
  pivot_longer(all_of(feature_names), names_to = "feature") |>
  mutate(feature = factor(feature, levels = feature_names))

features_long |>
  group_by(feature) |>
  summarise(mean = mean(value), sd = sd(value), min = min(value), max = max(value))

# Do writers differ? Feature distribution per writer (all digits and sessions pooled)
feature_by_writer <- ggplot(features_long, aes(factor(person_id), value)) +
  geom_boxplot(outlier.size = 0.2, outlier.alpha = 0.3, linewidth = 0.3, fill = "grey92") +
  facet_wrap(vars(feature), scales = "free_y", ncol = 2) +
  labs(title = "Image features by writer", x = "Writer", y = NULL)

save_figure(feature_by_writer, "features_by_writer", width = 12, height = 10)

# Do sessions shift a writer? One line per writer; parallel lines would mean a
# session effect shared by all writers, crossing lines a writer-specific one
feature_by_session <- features_long |>
  group_by(feature, person_id, session) |>
  summarise(value = mean(value), .groups = "drop") |>
  ggplot(aes(session, value, group = person_id)) +
  geom_line(colour = "#2a78d6", alpha = 0.5, linewidth = 0.4) +
  facet_wrap(vars(feature), scales = "free_y", ncol = 4) +
  labs(title = "Mean image features per writer across sessions",
       subtitle = "One line per writer, averaged over digits and replicates",
       x = "Session", y = NULL)

save_figure(feature_by_session, "features_by_session", width = 12, height = 6)

# Share of each feature's variance attributable to the design factors.
# The design is balanced, so the sums of squares add up without ambiguity.
variance_components <- c(
  "digit" = "Digit", "person" = "Writer", "session" = "Session",
  "digit:person" = "Writer x digit", "person:session" = "Writer x session",
  "digit:session" = "Digit x session", "digit:person:session" = "Writer x digit x session",
  "Residuals" = "Within cell (replicates)"
)

variance_shares <- features_long |>
  mutate(digit = factor(digit), person = factor(person_id), session = factor(session)) |>
  nest(.by = feature) |>
  mutate(anova = map(data, \(d) {
    fit <- summary(aov(value ~ digit * person * session, data = d))[[1]]
    tibble(term = trimws(rownames(fit)), share = fit[["Sum Sq"]] / sum(fit[["Sum Sq"]]))
  })) |>
  select(feature, anova) |>
  unnest(anova) |>
  mutate(component = factor(variance_components[term], levels = rev(variance_components)))

variance_shares |>
  select(feature, component, share) |>
  pivot_wider(names_from = feature, values_from = share) |>
  arrange(desc(component)) |>
  mutate(across(-component, \(x) round(100 * x, 1))) |>
  print(width = Inf)

variance_plot <- ggplot(variance_shares, aes(share, feature, fill = component)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.4) +
  scale_x_continuous(labels = scales::label_percent(), expand = expansion(mult = c(0, 0.02))) +
  scale_y_discrete(limits = rev) +
  scale_fill_manual(
    values = c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "grey75") |>
      set_names(variance_components),
    breaks = variance_components
  ) +
  labs(title = "Where does the variation in each feature come from?",
       x = "Share of total sum of squares", y = NULL, fill = NULL) +
  theme(panel.grid.major.y = element_blank())

save_figure(variance_plot, "feature_variance_shares", width = 10, height = 4.5)


# 5. PCA of raw pixels -------------------------------------------------------

# Within one digit: do images cluster by writer, and do sessions move those clusters?
pca_digit <- 2
pca_ids   <- meta |> filter(digit == pca_digit) |> pull(image_id)
pca       <- prcomp(pixels[pca_ids, ], rank. = 10)

round(100 * (pca$sdev^2 / sum(pca$sdev^2))[1:10], 1) # % variance explained

pca_scores <- meta |>
  filter(digit == pca_digit) |>
  bind_cols(as_tibble(pca$x[, 1:2]))

pca_plot <- ggplot(pca_scores, aes(PC1, PC2)) +
  geom_point(data = select(pca_scores, -person_id), colour = "grey88", size = 0.2) +
  geom_point(aes(colour = factor(session)), size = 0.5, alpha = 0.7) +
  facet_wrap(vars(person_id), ncol = 5, labeller = label_both) +
  scale_colour_manual(values = session_colours) +
  coord_fixed() +
  guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
  labs(title = paste0("Digit ", pca_digit, " - first two principal components by writer"),
       subtitle = "Grey: all writers. Coloured: this writer's images by session",
       colour = "Session")

save_figure(pca_plot, "pca_by_writer", width = 11, height = 9)


# 6. Checks for simulation artefacts and limitations ------------------------

# Pixel values: share of background, saturation, and how many grey levels are used
mean(images == 0)
mean(images == 255)
length(unique(as.vector(images)))

intensity_plot <- tibble(value = as.vector(images)) |>
  filter(value > 0) |>
  count(value) |>
  ggplot(aes(value, n)) +
  geom_col(width = 1, fill = "#2a78d6") +
  scale_y_continuous(labels = scales::label_comma()) +
  labs(title = "Distribution of non-zero pixel intensities", x = "Pixel value", y = "Pixels")

save_figure(intensity_plot, "pixel_intensities", width = 8, height = 4)

# Pixels that are never inked carry no information
sum(colSums(pixels) == 0)

# Exact duplicate images
sum(duplicated(pixels))

# Ink touching the image border suggests clipped digits
border_pixels <- pixel_row %in% c(1, image_size) | pixel_col %in% c(1, image_size)

features |>
  mutate(touches_border = rowSums(pixels[, border_pixels]) > 0) |>
  group_by(digit) |>
  summarise(share_touching_border = mean(touches_border))

# Empty or nearly empty images
features |> slice_min(ink_area, n = 5) |> select(image_id:replicate, ink_area, ink_total)
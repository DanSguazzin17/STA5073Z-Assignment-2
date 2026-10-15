# Plot defaults, colour scheme and saving shared across tasks

# Colour scheme: dark navy as the main colour, cyan as the contrasting accent,
# blues and greys for everything else
plot_colours <- c(
  navy       = "#0D386E",
  cyan       = "#00A9E0",
  blue       = "#2E6EB8",
  light_blue = "#A9CBEA",
  dark_grey  = "#4D4D4D",
  grey       = "#9A9A9A",
  light_grey = "#D6D6D6"
)

theme_assignment <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = "#EBEBEB", linewidth = 0.4),
      axis.text          = element_text(colour = plot_colours[["dark_grey"]]),
      axis.title         = element_text(colour = plot_colours[["dark_grey"]]),
      plot.title         = element_text(face = "bold", colour = plot_colours[["navy"]]),
      plot.subtitle      = element_text(colour = plot_colours[["dark_grey"]]),
      strip.text         = element_text(face = "bold", colour = plot_colours[["dark_grey"]]),
      legend.text        = element_text(colour = plot_colours[["dark_grey"]])
    )
}

save_figure <- function(plot, name, dir, width, height) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 150, bg = "white")
  if (interactive()) print(plot)
  invisible(plot)
}

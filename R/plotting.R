# Plot defaults and saving shared across tasks

theme_assignment <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title       = element_text(face = "bold")
    )
}

save_figure <- function(plot, name, dir, width, height) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(dir, paste0(name, ".png")), plot,
         width = width, height = height, dpi = 150, bg = "white")
  if (interactive()) print(plot)
  invisible(plot)
}

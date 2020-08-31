options(digits = 4, width = 84)
options(dplyr.print_min = 6, dplyr.print_max = 6)
options(cli.width = 85)

knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  fig.align = 'center',
  tidy = FALSE
)

library(ggplot2)

transparent_theme <- function() {
  thm <- 
    theme_bw() + 
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA), 
      plot.background = element_rect(fill = "transparent", colour = NA),
      legend.position = "top",
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA)
    )
  theme_set(thm)
}

transparent_theme()

tmwr_version <- function() {
  dt <- Sys.Date()
  ver <- read.dcf("DESCRIPTION")[1, "Version"]
  paste0("Version ", ver, " (", dt, ")")
}

pkg <- function(x) {
  cl <- match.call()
  x <- as.character(cl$x)
  paste0('<span class="pkg">', x, '</span>')
}

options(digits = 3, width = 55)

knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  fig.align = 'center',
  tidy = FALSE
)

options(dplyr.print_min = 6, dplyr.print_max = 6)

transparent_theme <- function() {
  library(ggplot2)
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

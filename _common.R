options(digits = 4, width = 84)
options(dplyr.print_min = 6, dplyr.print_max = 6)
options(cli.width = 85)

knitr::opts_chunk$set(
  comment = "#>",
  collapse = TRUE,
  fig.align = 'center',
  tidy = FALSE
)



theme_transparent <- function(...) {

  ret <- ggplot2::theme_bw(...)
  
  trans_rect <- ggplot2::element_rect(fill = "transparent", colour = NA)
  ret$panel.background  <- trans_rect
  ret$plot.background   <- trans_rect
  ret$legend.background <- trans_rect
  ret$legend.key        <- trans_rect
  
  ret$legend.position <- "top"
  
  ret
}

library(ggplot2)
theme_set(theme_transparent())

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

options(digits = 4, width = 84)
options(dplyr.print_min = 6, dplyr.print_max = 6)
options(cli.width = 85)
options(crayon.enabled = FALSE)

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

is_new_version <- function(x, path) {
  cl <- match.call()
  nm <- as.character(cl$x)
  if (!file.exists(path)) {
    return(TRUE)
  }
  load(path)
  prev <- get(nm)
  
  # parsnip model fits have an elapsed time and this will change from run-to-run.
  # We'll remove that to check for a new version. Same for workflows. 
  if (inherits(prev, "model_fit")) {
    x$elapsed <- NA
    prev$elapsed <- NA
  }
  if (workflows:::is_workflow(prev)) {
    x$fit$fit$elapsed <- NA
    prev$fit$fit$elapsed <- NA
  }
  
  res <- all.equal(x, prev)
  !isTRUE(res)
}


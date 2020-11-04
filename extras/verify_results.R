# These functions make sure that our results have not changed so that the interpreation
# in the text is not incorrect.

suppressPackageStartupMessages(library(digest))

verify_consistent_bo <- function(x) {
  # md5 generated on 2020-11-04
  expected <- "10f823b50ebe4032a01e414392619286"
  current <- digest::digest(x$.metrics)
  if (!identical(expected, current)) {
    rlang::abort(
      "These Bayesian optimization results don't match the previous values.")
  }
  invisible(NULL)
}


verify_consistent_sa <- function(x) {
  # md5 generated on 2020-11-04
  expected <- "5f04b0de490d2e04a3d3eb2b6d28f86e"
  current <- digest::digest(x$.metrics)
  if (!identical(expected, current)) {
    rlang::abort(
      "These simulated annealing results don't match the previous values.")
  }
  invisible(NULL)
}

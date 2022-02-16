# These functions make sure that our results have not changed so that the interpretation
# in the text is not incorrect.

verify_consistent_bo <- function(x) {
  # initial results generated on 2022-02-16
  load("RData/svm_bo_metrics.RData")
  bo_check <- all.equal(x, svm_bo_metrics, tolerance = 0.01)
  if (!isTRUE(bo_check)) {
    msg <- "These Bayesian optimization results don't match the previous values.:\n"
    msg <- paste0(msg, paste0(bo_check, collapse = "\n"))
    rlang::abort(msg)
  }
  invisible(NULL)
}


verify_consistent_sa <- function(x) {
  # initial results generated on 2022-02-16
  load("RData/svm_sa_metrics.RData")
  sa_check <- all.equal(x, svm_sa_metrics, tolerance = 0.01)
  if (!isTRUE(sa_check)) {
    msg <- "These simulated annealing results don't match the previous values.:\n"
    msg <- paste0(msg, paste0(sa_check, collapse = "\n"))
    rlang::abort(msg)
  }
  invisible(NULL)
}

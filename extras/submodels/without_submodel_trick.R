# remotes::install_github("tidymodels/parsnip@no-submodel-trick")
library(tidymodels)
library(tictoc)
library(mirai)

cores <- parallel::detectCores()

## -----------------------------------------------------------------------------

data(cells)
cells <- cells %>% select(-case)
set.seed(33)
cell_folds <- vfold_cv(cells)
roc_res <- metric_set(roc_auc)

## -----------------------------------------------------------------------------

c5_spec <-
  boost_tree(trees = tune()) %>%
  set_engine("C5.0") %>%
  set_mode("classification")

tic()
set.seed(2)
c5_spec %>%
  tune_grid(
    class ~ .,
    resamples = cell_folds,
    grid = data.frame(trees = 1:100),
    metrics = roc_res
  )
toc()

## -----------------------------------------------------------------------------

daemons(cores)

tic()
set.seed(2)
c5_spec %>%
  tune_grid(
    class ~ .,
    resamples = cell_folds,
    grid = data.frame(trees = 1:100),
    metrics = roc_res
  )
toc()

## -----------------------------------------------------------------------------

sessioninfo::session_info()

q("no")

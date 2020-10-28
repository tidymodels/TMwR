# remotes:::install_github("tidymodels/tune@monitor-execution-times")

library(tidymodels)
library(stringr)
library(lubridate)
library(doMC)

parse_logging_notes <- function(x) {
  require(dplyr)
  require(lubridate)
  require(stringr)
  msgs <-
    x %>%
    dplyr::select(starts_with("id"), .notes) %>%
    unnest(c(.notes)) %>%
    dplyr::mutate(.notes = map_chr(.notes, ~ str_remove_all(.x, "^internal: "))) %>%
    mutate(entries = map(.notes, ~ str_split(.x, ","))) %>%
    dplyr::select(-.notes) %>%
    unnest(c(entries)) %>%
    unnest(c(entries)) %>%
    mutate(entries = str_trim(entries, "left")) %>%
    mutate(split = str_split(entries, " at "),

           pid = map_chr(split, ~ str_remove(.x[[1]], "pid ")),

           event = map_chr(split, ~.x[[2]]),
           event_split = map(event, str_split, "model"),
           pre_proc = map_chr(event_split, ~ str_trim(.x[[1]][1], "right")),
           type = map_chr(event_split, ~ ifelse(length(.x[[1]]) > 1, "model", "preprocess")),
           time = map(split, ~ ymd_hms(.x[[3]])) %>% do.call('c', .),
           pre_proc = factor(pre_proc),
           pre_proc = reorder(pre_proc, time)
           ) %>%
    dplyr::select(starts_with("id"), event, time, pre_proc, type, pid)
  msgs
}

## -----------------------------------------------------------------------------

data(cells)
cells <- cells %>% select(-case)

set.seed(33)
cell_folds <- vfold_cv(cells)

roc_res <- metric_set(roc_auc)

mlp_rec <-
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = 30) %>%
  step_normalize(all_predictors())

mlp_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_engine("nnet", trace = 0) %>%
  set_mode("classification")

mlp_wflow <-
  workflow() %>%
  add_model(mlp_spec) %>%
  add_recipe(mlp_rec)

mlp_param <-
  mlp_wflow %>%
  parameters() %>%
  update(
    epochs = epochs(c(50, 200))
  )

registerDoMC(cores=12)

set.seed(99)
mlp_sfd_all <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = grid_regular(mlp_param, levels = 2),
    param_info = mlp_param,
    metrics = roc_res,
    control = control_grid(parallel_over = "everything")
  )

registerDoMC(cores=10)
set.seed(99)
mlp_sfd_rs <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = grid_regular(mlp_param, levels = 2),
    param_info = mlp_param,
    metrics = roc_res,
    control = control_grid(parallel_over = "resamples")
  )

## -----------------------------------------------------------------------------

times_all <-
  mlp_sfd_all %>%
  parse_logging_notes()

start_times_all <-
  times_all %>%
  group_by(pid) %>%
  summarize(first = min(time), .groups = "drop")

times_all <-
  full_join(times_all, start_times_all, by = "pid") %>%
  mutate(
    seconds = time - first,
    pid = factor(pid),
    pid = as.integer(pid),
    pid = paste("worker", format(pid))
  ) %>%
  dplyr::select(id, pre_proc, pid, seconds, type) %>%
  dplyr::rename(operation = type)

###

times_resamples <-
  mlp_sfd_rs %>%
  parse_logging_notes()

start_times_resamples <-
  times_resamples %>%
  group_by(pid) %>%
  summarize(first = min(time), .groups = "drop")

times_resamples <-
  full_join(times_resamples, start_times_resamples, by = "pid") %>%
  mutate(
    seconds = time - first,
    pid = factor(pid),
    pid = as.integer(pid),
    pid = paste("worker", format(pid))
  ) %>%
  dplyr::select(-event) %>%
  dplyr::rename(operation = type) 

save(times_all, times_resamples, file = "extras/parallel_times/logging_data.RData", version = 2)

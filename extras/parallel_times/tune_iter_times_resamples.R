#remotes::install_github("tidymodels/tune@monitor-execution-times")
# This will try to write to ~/tmp
library(tidymodels)
library(doParallel)
cl <- makePSOCKcluster(10)
registerDoParallel(cl)

options(width = 120)

data(cells)
cells <- cells %>% select(-case)

set.seed(6735)
folds <- vfold_cv(cells, v = 5)


cell_rec <-
  recipe(class ~ ., data = cells) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_ica(all_numeric_predictors(), num_comp = 30)

rf_mod <-
  rand_forest(mtry = tune(), min_n = tune(), trees = 50) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# Use a space-filling design with 7 points
set.seed(3254)
rf_res <- tune_grid(rf_mod, cell_rec, resamples = folds, grid = 7,
                     control = control_grid(parallel_over = "resamples"))


f_names <- list.files("~/tmp", pattern = "^time", full.names = TRUE)

timings <- NULL
for (i in f_names) {
  load(i)
  timings <- bind_rows(timings, res)
}


resamples_times <-
  timings %>%
  mutate(
    label = ifelse(mod_iter == 0, "preprocess", "model"),
    label = factor(label, levels = rev(c("preprocess", "model"))),
    pid = factor(format(pid)),
    pid = paste("worker", format(as.numeric(pid))),
    id_alt = paste(id, "/", pid)
  ) %>%
  arrange(pid, id, label)



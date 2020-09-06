library(tidymodels)

cores <- 1

data(cells)
cells <- cells %>% select(-case)

mlp_spec <-
  mlp(hidden_units = tune(),
      penalty = tune(),
      epochs = tune()) %>%
  set_engine("nnet", trace = 0) %>%
  set_mode("classification")

set.seed(33)
cell_folds <- vfold_cv(cells)

mlp_rec <-
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

mlp_wflow <-
  workflow() %>%
  add_model(mlp_spec) %>%
  add_recipe(mlp_rec)

mlp_param <-
  mlp_wflow %>%
  parameters() %>%
  update(epochs = epochs(c(50, 200)),
         num_comp = num_comp(c(0, 40)))

roc_res <- metric_set(roc_auc)

sfd_time <- system.time({
  set.seed(99)
  mlp_sfd_tune <-
    mlp_wflow %>%
    tune_grid(
      cell_folds,
      grid = 20,
      param_info = mlp_param,
      metrics = roc_res
    )
})


reg_time <- system.time({
  set.seed(99)
  mlp_reg_tune <-
    mlp_wflow %>%
    tune_grid(
      cell_folds,
      grid = mlp_param %>% grid_regular(levels = 3),
      param_info = mlp_param,
      metrics = roc_res
    )
})

times <- tibble::tibble(cores = cores, sfd = sfd_time[3], reg = reg_time[3])
save(times, file = paste0("mlp_", cores, format(Sys.time(), "_%Y_%m_%d_%H_%M_%S.RData")))

sessioninfo::session_info()

q("no")


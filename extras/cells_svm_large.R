library(tidymodels)
library(rayshader)
library(doMC)
registerDoMC(cores = parallel::detectCores(logical = TRUE))

## -----------------------------------------------------------------------------

data(cells)
cells <- cells %>% select(-case)
set.seed(33)
cell_folds <- vfold_cv(cells)
roc_res <- metric_set(roc_auc)

## -----------------------------------------------------------------------------

svm_rec <- 
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_predictors()) %>%
  step_normalize(all_predictors())

svm_spec <- 
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

svm_wflow <- 
  workflow() %>% 
  add_model(svm_spec) %>% 
  add_recipe(svm_rec)

svm_param <- 
  svm_wflow %>% 
  parameters() %>% 
  update(
    cost = cost(c(-10, 5)),
    rbf_sigma = rbf_sigma(c(-7, -1))
  )

## -----------------------------------------------------------------------------

large_grid <- grid_regular(svm_param, levels = 50)

set.seed(2)
svm_large <- 
  svm_wflow %>% 
  tune_grid(resamples = cell_folds, grid = large_grid, metrics = roc_res)

## -----------------------------------------------------------------------------

if (interactive()) {
  
  large_plot <-
    svm_large %>% 
    collect_metrics() %>% 
    ggplot(aes(x = rbf_sigma, y = cost)) + 
    geom_raster(aes(fill = mean)) + 
    scale_x_log10() + 
    scale_y_continuous(trans = "log2") +
    scale_fill_distiller(palette = "Blues") +
    theme_minimal() + 
    theme(legend.position = "bottom") + 
    guides(title.position = "bottom") + 
    labs(x = "rbf_sigma\n\n\n\n", title = NULL)
  
  plot_gg(
    large_plot,
    multicore = FALSE,
    raytrace = TRUE,
    width = 7,
    height = 7,
    scale = 300,
    windowsize = c(1400, 1400),
    zoom = 1,
    phi = 30,
    theta = 30
  )
  
}

## -----------------------------------------------------------------------------

sessioninfo::session_info()

## -----------------------------------------------------------------------------

save(svm_large, file = "RData/svm_large.RData")

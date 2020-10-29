library(tidymodels)
library(nycflights13)
library(doMC)
library(rlang)
library(xgboost)
library(vctrs)

## -----------------------------------------------------------------------------

num_resamples <- 5
num_grid <- 10
num_cores <- 1
preproc <- "light preprocessing"
par_method <- "resamples"

## -----------------------------------------------------------------------------

set.seed(123)

flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = as.Date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor) %>% 
  sample_n(4000)

## -----------------------------------------------------------------------------

flights_rec <- 
  recipe(arr_delay ~ ., data = flight_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors())

preproc_data <- 
  flights_rec %>% 
  prep() %>% 
  juice(all_predictors(), all_outcomes())

## -----------------------------------------------------------------------------

xgboost_spec <- 
  boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune(), 
             loss_reduction = tune(), sample_size = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost") 

## -----------------------------------------------------------------------------

if (preproc != "no preprocessing") {
  xgboost_workflow <- 
    workflow() %>% 
    add_recipe(flights_rec) %>% 
    add_model(xgboost_spec) 

  set.seed(33)
  bt <- bootstraps(flight_data, times = num_resamples)
} else {
  xgboost_workflow <- 
    workflow() %>% 
    add_variables(arr_delay, predictors = c(everything())) %>% 
    add_model(xgboost_spec) 
  
  set.seed(33)
  bt <- bootstraps(preproc_data, times = num_resamples)
}

## -----------------------------------------------------------------------------

set.seed(22)
xgboost_grid <- 
  xgboost_workflow %>% 
  parameters() %>% 
  update(trees = trees(c(100, 2000))) %>% 
  grid_max_entropy(size = num_grid)

## -----------------------------------------------------------------------------

if (num_cores > 1) {
  registerDoMC(cores=num_cores)
}

## -----------------------------------------------------------------------------

roc_res <- metric_set(roc_auc)

ctrl <- control_grid(parallel_over = par_method)

grid_time <- system.time({
  set.seed(99)
  xgboost_workflow %>%
    tune_grid(bt, grid = xgboost_grid, metrics = roc_res, control = ctrl)
})

## -----------------------------------------------------------------------------

times <- tibble::tibble(
  elapsed = grid_time[3],
  num_resamples = num_resamples,
  num_grid = num_grid,
  num_cores = num_cores,
  preproc = preproc,
  par_method = par_method
)


save(times, file = paste0("xgb_", num_cores, format(Sys.time(), "_%Y_%m_%d_%H_%M_%S.RData")))

sessioninfo::session_info()

if (!interactive()) {
  q("no")
}


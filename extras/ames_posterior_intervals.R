library(tidymodels)
library(mirai)
library(tidyposterior)
library(workflowsets)
library(rstanarm)
theme_set(theme_bw())

data(ames, package = "modeldata")

ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(123)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

cores <- parallel::detectCores()

daemons(cores)

## -----------------------------------------------------------------------------

set.seed(55)
ames_folds <- vfold_cv(ames_train, v = 10, repeats = 10)

lm_model <- linear_reg() %>% set_engine("lm")

rf_model <-
  rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# ------------------------------------------------------------------------------

basic_rec <-
  recipe(
    Sale_Price ~
      Neighborhood +
        Gr_Liv_Area +
        Year_Built +
        Bldg_Type +
        Latitude +
        Longitude,
    data = ames_train
  ) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

interaction_rec <-
  basic_rec %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_"))

spline_rec <-
  interaction_rec %>%
  step_ns(Latitude, Longitude, deg_free = 50)

preproc <-
  list(
    basic = basic_rec,
    interact = interaction_rec,
    splines = spline_rec,
    formula = Sale_Price ~
      Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude + Longitude
  )

models <- list(lm = lm_model, lm = lm_model, lm = lm_model, rf = rf_model)

four_models <-
  workflow_set(preproc, models, cross = FALSE)
four_models

posteriors <- NULL

for (i in 11:100) {
  if (i %% 10 == 0) {
    cat(i, "... ")
  }

  tmp_rset <- rsample:::df_reconstruct(ames_folds %>% slice(1:i), ames_folds)

  four_resamples <-
    four_models %>%
    workflow_map("fit_resamples", seed = 1, resamples = tmp_rset)

  ## -----------------------------------------------------------------------------

  rsq_anova <-
    perf_mod(
      four_resamples,
      prior_intercept = student_t(df = 1),
      chains = cores - 2,
      iter = 5000,
      seed = 2,
      cores = cores - 2,
      refresh = 0
    )

  rqs_diff <-
    contrast_models(
      rsq_anova,
      list_1 = "splines_lm",
      list_2 = "basic_lm",
      seed = 3
    ) %>%
    as_tibble() %>%
    mutate(label = paste(format(1:100)[i], "resamples"), resamples = i)

  posteriors <- bind_rows(posteriors, rqs_diff)

  rm(rqs_diff)
}

## -----------------------------------------------------------------------------

# ggplot(posteriors, aes(x = difference)) +
#   geom_histogram(bins = 30) +
#   facet_wrap(~label)
#
# ggplot(posteriors, aes(x = difference)) +
#   geom_line(stat = "density", trim = FALSE) +
#   facet_wrap(~label)

intervals <-
  posteriors %>%
  group_by(resamples) %>%
  summarize(
    mean = mean(difference),
    lower = quantile(difference, prob = 0.05),
    upper = quantile(difference, prob = 0.95),
    .groups = "drop"
  ) %>%
  ungroup() %>%
  mutate(
    mean = predict(loess(mean ~ resamples, span = .15)),
    lower = predict(loess(lower ~ resamples, span = .15)),
    upper = predict(loess(upper ~ resamples, span = .15))
  )

save(intervals, file = "RData/post_intervals.RData")

# ggplot(intervals,
#        aes(x = resamples, y = mean)) +
#   geom_path() +
#   geom_ribbon(aes(ymin = lower, ymax = upper), fill = "red", alpha = .1) +
#   labs(y = expression(paste("Mean difference in ", R^2)),
#        x = "Number of Resamples (repeated 10-fold cross-validation)")
#

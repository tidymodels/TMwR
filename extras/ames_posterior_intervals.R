library(tidymodels)
library(doMC)
library(tidyposterior)
library(rstanarm)
library(gganimate)
theme_set(theme_bw())

data(ames, package = "modeldata")

ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(833961)
ames_split <- initial_split(ames, prob = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <-  testing(ames_split)

crs <- parallel::detectCores()

registerDoMC(cores = crs)

## -----------------------------------------------------------------------------

set.seed(1352)
ames_folds <- vfold_cv(ames_train, v = 10, repeats = 10)

ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

no_splines <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_"))

with_splines <-
  no_splines %>%
  step_ns(Latitude, Longitude, deg_free = 20)

lm_wflow <-
  workflow() %>%
  add_recipe(ames_rec) %>%
  add_model(linear_reg() %>% set_engine("lm"))

basic_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train)

rf_model <-
  rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_wflow <-
  workflow() %>%
  add_recipe(basic_rec) %>%
  add_model(rf_model)

posteriors <- NULL

for(i in 10:100) {

  tmp_rset <- rsample:::df_reconstruct(ames_folds %>% slice(1:i), ames_folds)

  lm_no_splines <-
    lm_wflow %>%
    remove_recipe() %>%
    add_recipe(no_splines) %>%
    fit_resamples(resamples = tmp_rset)

  lm_with_splines <-
    lm_wflow %>%
    remove_recipe() %>%
    add_recipe(with_splines) %>%
    fit_resamples(resamples = tmp_rset)

  set.seed(598)
  rf_res <-
    rf_wflow %>%
    fit_resamples(resamples = tmp_rset)

  ## -----------------------------------------------------------------------------

  no_splines_rsq <-
    collect_metrics(lm_no_splines, summarize = FALSE) %>%
    filter(.metric == "rsq") %>%
    select(id, id2, `no splines` = .estimate)

  splines_rsq <-
    collect_metrics(lm_with_splines, summarize = FALSE) %>%
    filter(.metric == "rsq") %>%
    select(id, id2, `with splines` = .estimate)

  rf_rsq <-
    collect_metrics(rf_res, summarize = FALSE) %>%
    filter(.metric == "rsq") %>%
    select(id, id2, `random forest` = .estimate)

  rsq_estimates <-
    inner_join(no_splines_rsq, splines_rsq, by = c("id", "id2")) %>%
    inner_join(rf_rsq, by = c("id", "id2"))

  ames_two_models <-
    tmp_rset %>%
    bind_cols(rsq_estimates %>% arrange(id) %>% select(-id, -id2))

  ## -----------------------------------------------------------------------------

  rsq_anova <-
    perf_mod(
      ames_two_models,
      prior_intercept = student_t(df = 1),
      chains = crs - 2,
      iter = 5000,
      seed = 9791,
      cores = crs - 2,
      refresh = 0
    )

  rqs_diff <-
    contrast_models(rsq_anova,
                    list_1 = "with splines",
                    list_2 = "no splines",
                    seed = 6541) %>%
    as_tibble() %>%
    mutate(label = paste(format(1:100)[i], "resamples"), resamples = i)

  posteriors <- bind_rows(posteriors, rqs_diff)

  rm(rqs_diff)

}

## -----------------------------------------------------------------------------

ggplot(posteriors, aes(x = difference)) +
  geom_histogram(bins = 30) +
  facet_wrap(~label)

ggplot(posteriors, aes(x = difference)) +
  geom_line(stat = "density", trim = FALSE) +
  facet_wrap(~label)

# hist_anim <-
#   ggplot(posteriors, aes(x = difference)) +
#   geom_histogram(bins = 30, col = "white", fill = "red", alpha = 0.7) +
#   transition_states(
#     resamples,
#     transition_length = 2,
#     state_length = 1
#   ) +
#   enter_fade() +
#   exit_shrink() +
#   labs() +
#   theme_bw() +
#   labs(x = expression(paste("Posterior for mean difference in ", R^2,
#                             " (splines - no splines)")),
#        title = '{closest_state} resamples')
# anim <-
#   animate(
#     hist_anim,
#     nframes = 100,
#     fps = 20,
#     detail = 5,
#     width = 1000 * 2,
#     height = 500 * 2,
#     res = 200
#   )
#
# anim_save("~/tmp/posteriors.gif")


# ggplot(posteriors, aes(x = difference)) +
#   geom_line(stat = "density", trim = TRUE, adjust = 1.5) +
#   transition_states(
#     resamples,
#     transition_length = 2,
#     state_length = 1
#   ) +
#   enter_fade() +
#   exit_shrink() +
#   labs(title = '{closest_state} resamples') +
#   theme_bw()

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

ggplot(intervals,
       aes(x = resamples, y = mean)) +
  geom_path() +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "red", alpha = .1) +
  labs(y = expression(paste("Mean difference in ", R^2)),
       x = "Number of Resamples (repeated 10-fold cross-validation)")



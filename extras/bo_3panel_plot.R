collect_gp_results <- function(res, aqf = exp_improve(), num_vals = 100) {
  perf_res <- collect_metrics(res)
  iters <- max(perf_res$.iter)
  pset <- .get_tune_parameters(res)
  metrics <- .get_tune_metrics(res)
  metrics_data <- metrics_info(metrics)
  metrics_name <- metrics_data$.metric[1]
  perf_res <- dplyr::filter(perf_res, .metric == metrics_name)
  maximize <- metrics_data$direction[metrics_data$.metric == metrics_name] ==
    "maximize"

  grid <- grid_regular(pset, levels = num_vals)
  grid_scaled <- encode_set(grid, pset, as_matrix = TRUE)

  nm <- recipes::names0(iters, "gp_candidates_")
  file_name <- paste0(nm, ".RData")
  files <- file.path(tempdir(), file_name)
  has_files <- file.exists(files)
  if (any(!has_files)) {
    rlang::abort("No GP files were found in `tempdir()`")
  }

  tmp <- NULL
  for (i in 1:iters) {
    load(files[i])

    if (maximize) {
      current_best <- max(perf_res$mean[perf_res$.iter < i])
    } else {
      current_best <- min(perf_res$mean[perf_res$.iter < i])
    }
    gp_pred <- predict(gp_fit$fit, grid_scaled, se.fit = TRUE)
    candidates <-
      grid %>%
      bind_cols(tibble::tibble(.mean = gp_pred$mean, .sd = gp_pred$se))

    candidates <-
      bind_cols(
        candidates,
        predict(aqf, candidates, iter = i, maximize = maximize, current_best)
      ) %>%
      mutate(.iter = i)
    tmp <- dplyr::bind_rows(tmp, candidates)
    rm(candidates)
  }
  tmp
}

fmt_dcimals <- function(digits = 2) {
  function(x) format(x, digits = digits, scientific = FALSE)
}

mean_plot <- function(grid, object, iter) {
  grid <- dplyr::filter(grid, .iter == iter)
  res <- collect_metrics(object)
  initial <-
    res %>%
    dplyr::filter(.iter == 0) %>%
    dplyr::select(rbf_sigma, cost)
  existing <-
    res %>%
    dplyr::filter(.iter <= iter & .iter > 0) %>%
    dplyr::select(rbf_sigma, cost)

  grid %>%
    ggplot(aes(x = rbf_sigma, y = cost)) +
    geom_raster(aes(fill = .mean)) +
    scale_x_log10(labels = fmt_dcimals(2)) +
    scale_y_continuous(trans = "log2", labels = fmt_dcimals(2)) +
    geom_point(data = initial, col = "black", pch = 4, size = 5) +
    geom_point(data = existing, col = "black") +
    geom_point(
      data = grid %>% arrange(desc(objective)) %>% slice(1),
      size = 3,
      col = "green"
    ) +
    scale_fill_distiller(palette = "Blues") +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = 4),
      axis.text.x = element_text(size = 4)
    ) +
    labs(title = "predicted ROC AUC mean") +
    coord_fixed(ratio = 1 / 2.5)
}


sd_plot <- function(grid, object, iter) {
  grid <- dplyr::filter(grid, .iter == iter)
  res <- collect_metrics(object)
  initial <-
    res %>%
    dplyr::filter(.iter == 0) %>%
    dplyr::select(rbf_sigma, cost)
  existing <-
    res %>%
    dplyr::filter(.iter <= iter & .iter > 0) %>%
    dplyr::select(rbf_sigma, cost)
  grid %>%
    ggplot(aes(x = rbf_sigma, y = cost)) +
    geom_raster(aes(fill = -.sd)) +
    scale_x_log10(labels = fmt_dcimals(2)) +
    scale_y_continuous(trans = "log2", labels = fmt_dcimals(2)) +
    geom_point(data = initial, col = "black", pch = 4, size = 5) +
    geom_point(data = existing, col = "black") +
    geom_point(
      data = grid %>% arrange(desc(objective)) %>% slice(1),
      size = 3,
      col = "green"
    ) +
    scale_fill_distiller(palette = "Reds") +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = 4),
      axis.text.x = element_text(size = 4)
    ) +
    labs(title = "predicted ROC AUC std dev") +
    coord_fixed(ratio = 1 / 2.5)
}

improv_plot <- function(grid, object, iter) {
  grid <- dplyr::filter(grid, .iter == iter)
  res <- collect_metrics(object)
  initial <-
    res %>%
    dplyr::filter(.iter == 0) %>%
    dplyr::select(rbf_sigma, cost)
  existing <-
    res %>%
    dplyr::filter(.iter <= iter & .iter > 0) %>%
    dplyr::select(rbf_sigma, cost)
  grid %>%
    ggplot(aes(x = rbf_sigma, y = cost)) +
    geom_raster(aes(fill = objective)) +
    scale_x_log10(labels = fmt_dcimals(2)) +
    scale_y_continuous(trans = "log2", labels = fmt_dcimals(2)) +
    geom_point(data = initial, col = "black", pch = 4, size = 5) +
    geom_point(data = existing, col = "black") +
    geom_point(
      data = grid %>% arrange(desc(objective)) %>% slice(1),
      size = 3,
      col = "green"
    ) +
    scale_fill_gradientn(
      colours = rev(scales::brewer_pal(palette = "BuPu")(4))
    ) +
    theme(
      legend.position = "none",
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(size = 8)
    ) +
    labs(title = "predicted expected improvement") +
    coord_fixed(ratio = 1 / 2.5)
}

transparent_theme <- function() {
  thm <-
    theme_bw() +
    theme(
      panel.background = element_rect(fill = "transparent", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      legend.position = "top",
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.key = element_rect(fill = "transparent", colour = NA),
      plot.title = element_text(hjust = 0.5)
    )
  theme_set(thm)
}

make_bo_animation <- function(grid, object) {
  layout <- "
  12
  33
  33
"
  require(patchwork)
  num_iter <- max(collect_metrics(object)$.iter)

  files <- purrr::map_chr(1:num_iter, ~ tempfile(pattern = "bo_plot_"))
  iter_chr <- format(1:num_iter)

  all_plots <- vector(mode = "list", length = num_iter)

  for (i in 1:num_iter) {
    iter_lab <- paste("Iteration", iter_chr[i], "of", length(files))

    .mean <- mean_plot(grid, object, i)
    .sd <- sd_plot(grid, object, i)
    .impr <- improv_plot(grid, object, i)

    transparent_theme()

    print(
      .mean +
        .sd +
        .impr +
        plot_layout(design = layout) +
        plot_annotation(
          iter_lab,
          theme = theme(plot.title = element_text(hjust = 0.5))
        )
    )
  }
  invisible(NULL)
}

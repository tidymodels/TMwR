sa_2d_plot <- function(sa_obj, history, large_sa, path = tempdir()) {
  params <-
    sa_obj %>%
    collect_metrics() %>%
    select(.iter, cost, rbf_sigma) %>%
    arrange(.iter)
  
  init <-
    params %>%
    filter(.iter == 0)
  
  ## -----------------------------------------------------------------------------
  
  svm_roc <-
    large_sa %>%
    collect_metrics()
  
  large_plot <-
    svm_roc %>%
    ggplot(aes(x = rbf_sigma, y = cost)) +
    geom_raster(aes(fill = mean), show.legend = FALSE) +
    scale_x_log10() +
    scale_y_continuous(trans = "log2") +
    scale_fill_distiller(palette = "Blues") +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      legend.key.width = grid::unit(2, "cm"),
      plot.title = element_text(hjust = 0.5)
    ) +
    guides(title.position = "bottom") +
    labs(x = "rbf_sigma\n\n\n\n", title = "ROC AUC surface") +
    coord_fixed(ratio = 1/2.5)
  
  base_plot <-
    large_plot +
    geom_point(data = init, pch = 4, cex = 4)
  
  ## -----------------------------------------------------------------------------
  
  num_init <- nrow(init)
  num_iter <- max(history$.iter)
  
  nms <- purrr::map_chr(1:nrow(history), ~ tempfile())
  
  for (i in (num_init + 1):nrow(history)) {
    current_iter <- history$.iter[i]
    current_res <- current_param_path(history, current_iter)
    current_best <- current_res %>% dplyr::filter(results == "new best")
    
    ttl <- paste0("Iteration ", current_iter)
    
    text_just <-
      case_when(
        history$results[i] == "restart from best"  ~0.00,
        history$results[i] == "discard suboptimal" ~ 0.25,
        history$results[i] == "accept suboptimal"  ~ 0.50,
        history$results[i] == "better suboptimal"  ~ 0.75,
        history$results[i] == "new best"           ~ 1.00
      )
    
    tmp <- history
    tmp$results <- gsub(" suboptimal", "\nsuboptimal",  tmp$results)
    tmp$results <- gsub(" best", "\nbest",  tmp$results)
    
    new_plot <-
      base_plot +
      geom_point(
        data = current_res %>% slice(n()),
        size = 3,
        col = "green"
      ) +
      geom_path(
        data = current_res,
        alpha = .5,
        arrow = arrow(length = unit(0.15, "inches"))
      ) +
      ggtitle(ttl, subtitle = tmp$results[i]) +
      theme(plot.subtitle = element_text(hjust = text_just))
    
    if (nrow(current_best) > 0) {
      new_plot <-
        new_plot +
        geom_point(data = current_best, size = 1/3)
    }
    print(new_plot)
  }
  invisible(NULL)
}

current_param_path <- function(x, iter) {
  x <-
    x %>%
    dplyr::filter(.iter <= iter)
  ind <- nrow(x)
  param_path <- ind
  while(length(ind) > 0) {
    ind <- which(x$.config == x$.parent[ind])
    param_path <- c(param_path, ind)
  }
  x %>% dplyr::slice(rev(param_path))
}

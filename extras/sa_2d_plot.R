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
    geom_point(data = top_n(svm_roc, 1, mean)) + 
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
    ind <- get_path(history, i)
    
    if (history$new_best[i]) {
      ttl <- paste0("Iteration ", history$.iter[i], " (new global best)")
    } else {
      ttl <- paste0("Iteration ", history$.iter[i], " (", history$results[i], ")")
    }
    
    new_plot <-
      base_plot +
      geom_point(
        data = params %>% slice(ind) %>% slice(n()),
        size = 3,
        col = "green"
      ) +
      geom_path(
        data = params %>% slice(ind),
        alpha = .5,
        arrow = arrow(length = unit(0.15, "inches"))
      ) +
      ggtitle(ttl)
    
    
    print(new_plot)
    
  }

  invisible(NULL)
}


# walks along the history and finds points with a new global maximum
get_new_best <- function(x) {
  n <- nrow(x)
  ind_search <- (1:n)[x$.iter == 1]
  
  current_best <- max(x$mean[x$.iter == 0])
  
  last_best <- ind_search
  x$new_best <- FALSE
  for(i in ind_search:n) {
    if (x$results[i] == "improvement") {
      if (x$mean[i] >= current_best) {
        x$new_best[i] <- TRUE
        current_best <- x$mean[i]
      }
    }
  }
  
  x %>% 
    select(-random, -accept, -n, -std_err, -.metric, -global_best)
}

# For each new point, determine the previus point that was perturbed to
# make the new value. 
get_parent <- function(x) {
  n <- nrow(x)
  ind_search <- (1:n)[x$.iter == 1]
  
  last_best <- last_accepted <- ind_search
  x$parent <- x$last_accepted <- x$last_best <- which.max(x$mean[x$.iter == 0])
  
  for(i in ind_search:n) {
    if (x$new_best[i]) {
      x$last_best[i] <- i
    } else {
      x$last_best[i] <- max(x$last_best[ind_search:(i-1)], na.rm = TRUE)
    }
    if (x$results[i - 1] != "discard") {
      x$last_accepted[i] <- i - 1
    } else {
      x$last_accepted[i] <- max(x$last_accepted[ind_search:(i-1)], na.rm = TRUE)
    }
  }
  for(i in (ind_search + 1):n) {
    if (x$results[i - 1] == "discard") {
      x$parent[i] <- max(x$last_accepted[ind_search:(i-1)], na.rm = TRUE)
    }
    if (x$results[i - 1] == "restart") {
      x$parent[i] <- max(x$last_best[ind_search:(i-1)], na.rm = TRUE)
    }    
    if (!(x$results[i - 1] %in% c("discard", "restart"))) {
      x$parent[i] <- i - 1
    }      
  }
  x
}

# For a specific point, find the points that were used to create it and its
# parents. 
get_path <- function(x, i) {
  n <- nrow(x)
  ind_search <- (1:n)[x$.iter == 1]
  
  res <- i
  while (i > ind_search) {
    res <- c(res, x$parent[i])
    i <- x$parent[i]
  }
  res <- sort(unique(res))
  res <- c(which.max(x$mean[x$.iter == 0]), res)
  res
}



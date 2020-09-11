collect_gp_results <- function(x) {
  iters <- max(collect_metrics(x)$.iter)
  nm <- recipes::names0(iters, "gp_candidates_")
  file_name <- paste0(nm, ".RData")
  files <- file.path(tempdir(), file_name)
  has_files <- file.exists(files)
  if (any(!has_files)) {
    
  }
  tmp <- NULL
  for(i in files) {
    load(i)
    tmp <- dplyr::bind_rows(tmp, candidates)
    rm(candidates)
  }
  tmp
}

fmt_dcimals <- function(digits = 2){
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
    geom_point(data =  grid %>% arrange(desc(objective)) %>% slice(1), 
               size = 3, col = "green") + 
    scale_fill_distiller(palette = "Blues") +
    theme(legend.position = "none") +
    labs(title = "predicted ROC AUC mean") + 
    coord_fixed(ratio = 1/2.5)
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
    geom_point(data =  grid %>% arrange(desc(objective)) %>% slice(1), 
               size = 3, col = "green") + 
    scale_fill_distiller(palette = "Reds") +
    theme(legend.position = "none") +
    labs(title = "predicted ROC AUC std dev") + 
    coord_fixed(ratio = 1/2.5)
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
    geom_point(data =  grid %>% arrange(desc(objective)) %>% slice(1), 
               size = 3, col = "green") + 
    scale_fill_gradientn(colours = rev(scales::brewer_pal(palette = "BuPu")(4))) +
    theme(legend.position = "none") +
    labs(title = "predicted expected improvement") + 
    coord_fixed(ratio = 1/2.5)
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
  design <- "
  11##  
  1133 
  2233 
  22## 
"
  require(patchwork)
  num_iter <- max(collect_metrics(object)$.iter)
  
  files <- purrr::map_chr(1:num_iter, ~ tempfile(pattern = "bo_plot_"))
  iter_chr <- format(1:num_iter)
  
  all_plots <- vector(mode = "list", length = num_iter)
  
  for(i in 1:num_iter) {
    
    iter_lab <- paste("Iteration", iter_chr[i], "of", length(files))
    
    .mean <- mean_plot(grid, object, i)
    .sd   <- sd_plot(grid, object, i)
    .impr <- improv_plot(grid, object, i)
    

    transparent_theme()
    
    print(
      (.mean + .sd) / .impr + 
      plot_layout(design = design) + 
      plot_annotation(iter_lab, theme = theme(plot.title = element_text(hjust = 0.5)))
    )
    
  }
  invisible(NULL)
}


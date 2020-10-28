library(tidymodels)
library(lubridate)

## -----------------------------------------------------------------------------

get_date <- function(x) {
   x <- basename(x)
   x <- strsplit(x, "_")
   x <- map(x, ~ .x[3:8])
   x <- map(x, ~ gsub("\\.RData", "", .x))
   x <- map_chr(x, paste0, collapse = "-")
   ymd_hms(x)
}

get_times <- function(x) {
   load(x)
   res <- 
      times %>% 
      mutate(date = get_date(x))

   res
}

## -----------------------------------------------------------------------------

rdata <-
   list.files(path = ".",
              pattern = "\\.RData",
              full.names = TRUE)

all_times <-  map_dfr(rdata, get_times) 

seq <- 
   all_times %>% 
   filter(num_cores == 1) %>% 
   dplyr::rename(seq_time = elapsed) %>% 
   select(-num_cores, -date) 

times <- 
   full_join(all_times, seq, 
             by = c("num_resamples", "num_grid", "preproc", "par_method")) %>% 
   mutate(
      time_per_fit = elapsed/(num_grid * num_resamples),
      speed_up = seq_time/elapsed,
      preprocessing = gsub(" preprocessing", "", preproc),
      preprocessing = ifelse(preprocessing == "no", "none", preprocessing),
      preprocessing = factor(preprocessing, levels = c("none", "light", "expensive"))
   )

if (interactive()) {
   ggplot(times, aes(x = num_cores, y = elapsed, col = preproc)) + 
      geom_point() + 
      geom_line() +
      facet_wrap(~ par_method) + 
      labs(x = "Number of Workers", y = "Time per Submodel")

   
   times %>% 
      filter(preprocessing == "none") %>% 
      ggplot(aes(x = num_cores, y = speed_up, col = preprocessing)) + 
      geom_abline(lty = 1) + 
      geom_point() + 
      geom_line() +
      facet_wrap(~ par_method) + 
      coord_obs_pred() +
      labs(x = "Number of Workers", y = "Speed-up", 
           title = "5 resamples, 10 grid points") + 
      theme_bw() + 
      theme(legend.position = "top")
   
   times %>% 
      filter(preprocessing != "expensive") %>% 
      ggplot(aes(x = num_cores, y = speed_up, col = preprocessing)) + 
      geom_abline(lty = 1) + 
      geom_point() + 
      geom_line() +
      facet_wrap(~ par_method) + 
      coord_obs_pred() +
      labs(x = "Number of Workers", y = "Speed-up", 
           title = "5 resamples, 10 grid points") + 
      theme_bw() + 
      theme(legend.position = "top")

   
   ggplot(times, aes(x = num_cores, y = speed_up, col = preprocessing)) + 
      geom_abline(lty = 1) + 
      geom_point() + 
      geom_line() +
      facet_wrap(~ par_method) + 
      coord_obs_pred() +
      labs(x = "Number of Workers", y = "Speed-up", 
           title = "5 resamples, 10 grid points") + 
      theme_bw() + 
      theme(legend.position = "top")
   
}

save(times, file = "RData/xgb_times.RData")

# r_files <- list.files(path = ".", pattern = "R$")
# r_files <- r_files[r_files != "collect.R"]
# r_files <- r_files[r_files != "template.R"]
# r_files <- paste0("R CMD BATCH --vanilla ", r_files, "\nsleep 20\n")
# cat(sample(r_files), sep = "")

q("no")


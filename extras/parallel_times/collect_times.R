library(tidymodels)
library(lubridate)

rdata <-
  list.files(path = "extras/parallel_times/",
             pattern = "\\.RData",
             full.names = TRUE)

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
  times %>% 
    mutate(date = get_date(x))
}

all_times <- 
  map_dfr(rdata, get_times) %>% 
  pivot_longer(cols = c(sfd, reg), names_to = "grid", values_to = "time") %>% 
  group_by(cores, grid) %>% 
  summarize(time = mean(time), .groups = "drop") %>% 
  ungroup()

seq <- 
  all_times %>% 
  filter(cores == 1) %>% 
  dplyr::rename(seq_time = time) %>% 
  select(-cores)

times <- 
  full_join(all_times, seq, by = c("grid")) %>% 
  mutate(
    num_sub_models = ifelse(grid == "sfd", 20, 81),
    time_per_fit = time/(num_sub_models * 10),
    speed_up = seq_time/time,
    `Grid Type` = ifelse(grid == "sfd", "Space-Filling", "Regular")
  )


if (interactive()) {
  ggplot(times, aes(x = cores, y = time_per_fit, col = `Grid Type`)) + 
    geom_point() + 
    geom_line() +
    labs(x = "Number of Workers", y = "Time per Submodel")
  
  ggplot(times, aes(x = cores, y = speed_up, col = `Grid Type`)) + 
    geom_abline() + 
    geom_point() + 
    geom_line() + 
    coord_obs_pred() +
    labs(x = "Number of Workers", y = "Speed-up")
}

save(times, file = "RData/mlp_times.RData")


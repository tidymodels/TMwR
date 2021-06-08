library(tidymodels)
library(RWeka)
library(janitor)

dry_beans <- 
  read.arff(url("https://www.muratkoklu.com/datasets/vtdhnd02.php")) %>% 
  dplyr::rename(AspectRatio = AspectRation) %>% 
  clean_names() %>% 
  as_tibble() %>% 
  mutate(class = tolower(as.character(class)),
         class = factor(class))

names(dry_beans) <- gsub("([1-4]$)", "_\\1", names(dry_beans), perl = TRUE)

save(dry_beans, file = "RData/dry_beans.RData", compress = "xz", version = 2)


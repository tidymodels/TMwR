library(tidyverse)
library(affyPLM)
library(AmpAffyExample)

# ------------------------------------------------------------------------------

data(AmpData)
sampleNames(AmpData) <- c("N1","Good Quality","Poor Quality","A1","A2","A3")

Pset1 <- fitPLM(AmpData)

# ------------------------------------------------------------------------------

# Take from the image method for PLMset objevts 
pm.index <- unlist(affy::indexProbes(Pset1, "pm", row.names(coefs(Pset1))))
rows <-  Pset1@nrow
cols <-  Pset1@ncol
pm.x.locs <- pm.index %% rows
pm.x.locs[pm.x.locs == 0] <- rows
pm.y.locs <- pm.index %/% rows + 1

# ------------------------------------------------------------------------------

plm_resids <- 
  tibble::as_tibble(Pset1@residuals$PM.resid) %>% 
  mutate(
    probe = rownames(Pset1@residuals$PM.resid),
    x = pm.x.locs,
    y = pm.y.locs
  ) %>% 
  pivot_longer(cols = c(1:6), names_to = "Sample", values_to = "Intensity") %>% 
  dplyr::filter(Sample %in% c("Good Quality", "Poor Quality"))

# ------------------------------------------------------------------------------

save(plm_resids, file = "RData/plm_resids.RData")

  
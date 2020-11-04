nonlin_function <- function(x, error = TRUE) {
  # use the ames spline curve for Longitude just because I think that it's 
  # cool
  data(ames, package = "modeldata")
  rec <- 
    recipe(Sale_Price ~ Longitude, data = ames) %>% 
    step_log(Sale_Price, skip = TRUE) %>% 
    step_range(Longitude) %>% 
    prep()
  
  
  # use the ames longitude pattern since I like it
  f <- lm(log10(Sale_Price) ~ splines::ns(Longitude, df = 12), data = juice(rec))
  p <- predict(f, newdata = data.frame(Longitude = x), se.fit = TRUE)
  err <- p$se.fit
  if (!error) {
    err <- 0
  }
  rnorm(1, mean = p$fit, sd = err)
}

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
  res <- rnorm(1, mean = p$fit, sd = err)
  # convert to a R^2-like value
  res <- (8 * res)/10
  res <- max(res, 0)
  res <- min(res, 1)
  res
}

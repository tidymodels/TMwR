



# A tidyverse primer {#tidyverse-primer}

## Principles

What does it mean to be "tidy" (distinguish tidy data vs tidy interfaces etc. )


## Code

Things that I think that we'll need summaries of:

 * strategies: variable specification, pipes (with data or other first arguments), conflicts and using namespaces, splicing, non-standard evaluation, 

 * tactics: `select`, `bind_cols`, `tidyselect`, `slice`,  `!!` and `!!!`, `...` for passing arguments, tibbles, joins, `nest`/`unnest`, `group_by`


## A review of base R modeling syntax {#r-review}




This book is about software, specifically R syntax for creating models. Before descrbing how tidy principles can be used in data analysis, it makes sense to show how models are created and utilized using traditional base R code. This section is a brief illustration of the those conventions. It is not exhaustive but provides readers uninitiated to R ideas about the basic motifs that are commonly used. 

The S language, on which R is based, has had a rich data analysis environment since the publication of @WhiteBook (commonly known as The White Book). This version of S introduced standard infrastructure components, such as symbolic model formulae, model matrices, data frames, as well as the standard object-oriented programming methods for data analysis. Much of these implementations have not substantively changes since then.  

To demonstrate the fundamentals, experimental data from @mcdonald2009 (by way of @mangiafico2015) are used. These data relate how the ambient temperature related to the rate of cricket chirps per minute. Data were collected for two species: _O. exclamationis_ and _O. niveus_. The data are contained in a data frame called `crickets` that contains a total of 31 data points. These data are shown via a `ggplot` graph. 


```r
library(ggplot2)

names(crickets)
#> [1] "species" "temp"    "rate"

# Plot the temperature on the x-axis, the chirp rate on the y-axis. The plot
# elements will be colored differently for each species:
ggplot(crickets, aes(x = temp, y = rate, col = species)) + 
  # Plot points for each data point and color by species
  geom_point() + 
  # Show a simple linear model fit created separately for each species:
  geom_smooth(method = lm, se = FALSE) + 
  labs(x = "Temperature (C)", y = "Chirp Rate (per minute)")
```

<img src="figures/tidyverse-cricket-plot-1.svg" width="100%" style="display: block; margin: auto;" />
 
The data show fairly linear trends for each species. For a given temperature, _O. exclamationis_ appears to have more chirps than the other species. For an inferential model, the researchers might have specified the following null hypotheses prior to seeing the data:

 * Temperature has no affect on the chirp rate (denoted as hypothesis #1)

 * There are no differences between the species in terms of chirp rate. 

There may be some scientific rationale for being able to predict the chirp rate but the focus here will be on inference.

To fit an ordinary linear model, the `lm()` function is commonly used. The important arguments to this function are a model formula and a data frame that contains the data The formula is _symbolic_. For example, the simple formula:

```r
rate ~ temp
```
states that the chirp rate is the outcome (since it is on the left-hand side of the tilde `~`) and that the temperature values are the predictor^[Most model functions implicitly add an intercept column.]. Suppose the data contained the time of day in which the measurements were obtained in a column called `time`. The formula

```r
rate ~ temp + time
```

would not add the time and temperature values together. This formula would symbolically represent that temperature and time should be added as a separate _main effects_ to the model. Main effects are model terms that contain a single predictor variable. 

There are no time measurements in these data but the species can be added to the model in the same way: 

```r
rate ~ temp + species
```

Species is not a quantitative variable; in the data frame, it is represented as a factor column with levels `"O. exclamationis"` and `"O. niveus"`. The vast majority of model functions cannot operate on non-numeric data. For species, the model needs to _encode_ the species data into a numeric format. The most common approach is to use indicator variables (also known as "dummy variables") in place of the original qualitative values. In this instance, since species has two possible values, the model formula will automatically encode this column as numeric by adding a new column that has a value of zero when the species is `"O. exclamationis"` and a value of one when the data correspond to `"O. niveus"`. The underlying formula machinery will automatically convert these data for the data used to create the model as well as for any new data points (for example, when the model is used for prediction). 

Suppose there were five species. The model formula would automatically add _four_ additional binary columns that are binary indicators for four of the species. The _reference level_ of the factor (i.e., the first level) is always left out of the predictor set. The idea is that, if you know the values of the four indicator variables, the value of the species can be determined. 

The model formula shown above creates a model where there are different y-intercepts for each species. It is a reasonable supposition that the slopes of the regression lines could be different for each species. To accommodate this structure, an _interaction_ term can be added to the model. This can be specified in a few different ways, the most basic uses the colon:

```r
rate ~ temp + species + temp:species

# A shortcut can be used to expand all interactions containing
# interactions with two variables:
rate ~ (temp + species)^2
```

In addition to the convenience of automatically creating indicator variables, the formula offers a few other niceties: 

* _In-line_ functions can be used in the formula. For example, if the natural log of the temperate were used, the formula `rate ~ log(temp)` could be used. Since the formula is symbolic by default, literal math can be done to the predictors using the identity function `I()`. For example, to use Fahrenheit units, the formula could be `rate ~ I( (temp * 9/5) + 32 )` to make the conversion.

* R has many functions that are useful inside of formulas. For example, `poly(x, 3)` would create linear, quadratic, and cubic terms for `x` to the model as main effects. Also, the `splines` package has several functions to create nonlinear spline terms in the formula. 

* For data sets where there are many predictors, the period shortcut is available. The period represents main effects for all of the columns that are not on the left-hand side of the tilde. For example, using `~ (.)^3` would create main effects as well as all two- and three-variable interactions to the model. 

For the initial data analysis, the two-factor interaction model is used. In this book, the suffix `_fit` is used for R objects for fitted models. 


```r
interaction_fit <-  lm(rate ~ (temp + species)^2, data = crickets) 

# To print a short summary of the model:
interaction_fit
#> 
#> Call:
#> lm(formula = rate ~ (temp + species)^2, data = crickets)
#> 
#> Coefficients:
#>           (Intercept)                   temp       speciesO. niveus  
#>               -11.041                  3.751                 -4.348  
#> temp:speciesO. niveus  
#>                -0.234
```

This output is a little hard to read. For the species indicator variables, R mashes the variable name (`species`) together with the factor level (`O. niveus`) with no delimiter. 

Before going into any results for this model, the fit should be assessed using diagnostic plots. The `plot()` method for `lm` objects can be used. It produces a set of four plots for the object, each showing different aspects of the fit. Two plots are shown here:


```r
# Place two plots next to one another:
par(mfrow = c(1, 2))

# Show residuals vs predicted values:
plot(interaction_fit, which = 1)

# A normal quantile plot on the residuals:
plot(interaction_fit, which = 2)
```

<img src="figures/tidyverse-interaction-plots-1.svg" width="100%" style="display: block; margin: auto;" />

These appear reasonable enough to conduct inferential analysis. 

From a technical standpoint, R is _lazy_. Model fitting functions typically compute the minimum possible quantities. For example, there may be interest in the coefficient table for each model term. This is not automatically computed but is instead computed via the `summary()` method. 

Our second order of business is to assess if the inclusion of the interaction term is necessary. The most appropriate approach for this model is to re-compute the model without the interaction term and use the `anova()` method. 


```r
# Fit a reduced model:
main_effect_fit <-  lm(rate ~ temp + species, data = crickets) 

# Compare the two:
anova(main_effect_fit, interaction_fit)
#> Analysis of Variance Table
#> 
#> Model 1: rate ~ temp + species
#> Model 2: rate ~ (temp + species)^2
#>   Res.Df  RSS Df Sum of Sq    F Pr(>F)
#> 1     28 89.3                         
#> 2     27 85.1  1      4.28 1.36   0.25
```

The results of the statistical test generates a p-value of 0.3. This value implies that there is a lack of evidence for the alternative hypothesis that the the interaction term is needed by the model. For this reason, further analysis will be conducted on the model without the interaction. 

Residual plots should be re-assessed to make sure that our theoretical assumptions are valid enough to trust the p-values produced by the model (not shown but spoiler alert: they are). 

The `summary()` method is used to inspect the coefficients, standard errors, and p-values of each model term: 

```r
summary(main_effect_fit)
#> 
#> Call:
#> lm(formula = rate ~ temp + species, data = crickets)
#> 
#> Residuals:
#>    Min     1Q Median     3Q    Max 
#> -3.013 -1.130 -0.391  0.965  3.780 
#> 
#> Coefficients:
#>                  Estimate Std. Error t value Pr(>|t|)    
#> (Intercept)       -7.2109     2.5509   -2.83   0.0086 ** 
#> temp               3.6028     0.0973   37.03  < 2e-16 ***
#> speciesO. niveus -10.0653     0.7353  -13.69  6.3e-14 ***
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
#> 
#> Residual standard error: 1.79 on 28 degrees of freedom
#> Multiple R-squared:  0.99,	Adjusted R-squared:  0.989 
#> F-statistic: 1.33e+03 on 2 and 28 DF,  p-value: <2e-16
```

From these values, the chirp rate for each species increases by 3.6 chirps as the temperature increases by a single degree. This term shows strong statistical significance as evidenced by the p-value.  The species term has a value of -10.07. This indicates that, across all temperature values, _O. niveus_ is a  chirp rate that is about 10 fewer chirps per minute. Similar to the temperature term, the species effect is associated with a very small p-value.  

The only issue in this analysis is the intercept value. It indicates that at 0 C, there are -7.21 chirps per minute. While this is unreasonable, the data only go as low as 17.2 C and interpreting the model at 0 C would be an _extrapolation_. This would be a bad idea. That said, the model fit is good within the _applicable range_ of the temperature values and the conclusions should be limited to the observed temperature range. 

If there were a need to estimate the chirp rate at a temperature that was not observed in the experiment, the `predict()` method would be used. It takes the model object and a data frame of new values for prediction. For example, the model estimates the chirp rate for _O. exclamationis_ for temperatures between 15 C and 20 C can be computed via:


```r
new_values <- data.frame(species = "O. exclamationis", temp = 15:20)
predict(main_effect_fit, new_values)
#>    1    2    3    4    5    6 
#> 46.8 50.4 54.0 57.6 61.2 64.8
```

Note that the non-numeric value of `species` is given to the predict method (as opposed to the binary indicator variable).  

While this analysis has obviously not been an exhaustive demonstration of R's modeling capabilities, it does highlight some of the major features: 

 * The language has an expressive syntax for specifying model terms for simple and fairly complex models.

 * For formula method has many conveniences for modeling that are also applied to new data when predictions are generated. 

 * There are numerous helper functions (e.g., `anova()`, `summary()` and `predict()`) that are used to conduct specific calculations after the fitted model is created. 

Finally, as previously mentioned, this framework was devised in 1992. Most of the ideas and methods above were developed in that period and have remained remarkably relavant to this day. It highlights that the S language and, by extension R, has been designed as a language for data analysis since its inception.  




## Why tidiness is important for modeling

One of the strengths of R is that it encourages developers to create a user-interface that fits their needs.  As an example, here are three common methods for creating a scatter plot of two numeric variables residing in a data frame called `plot_data`:


```r
plot(plot_data$x, plot_data$y)

library(lattice)
xyplot(y ~ x, data = plot_data)

library(ggplot2)
ggplot(plot_data, aes(x = y, y = y)) + geom_point()
```

In this case, separate groups of developers devised distinct interfaces for the same task. Each has advantages and disadvantages. 

In comparison, the _Python Developer's Guide_ espouses the notion that, when approaching a problem:

> "There should be one-- and preferably only one --obvious way to do it."

The advantage of R's diversity of interfaces is that it it can evolve over time and fit different types of needs for different users. 

Unfortunately, some of the syntactical diversity is due to a focus on the developer's needs instead of the needs of the end-user. For example, one issue with some existing methods in base R is that the manner in which some data are stored may not be the most useful. For example, in Section \@ref(r-review) the results of linear model were saved: 


```r
main_effect_fit
#> 
#> Call:
#> lm(formula = rate ~ temp + species, data = crickets)
#> 
#> Coefficients:
#>      (Intercept)              temp  speciesO. niveus  
#>            -7.21              3.60            -10.07
```

The `summary()` method was used to print the results of the model fit, including a table with parameter values, their uncertainty estimates, and p-values. These particular results can also be saved:


```r
model_res <- summary(main_effect_fit)
# The model coefficient table is accessible via the `coef`
# method.
param_est <- coef(model_res)
class(param_est)
#> [1] "matrix"
param_est
#>                  Estimate Std. Error t value Pr(>|t|)
#> (Intercept)         -7.21     2.5509   -2.83 8.58e-03
#> temp                 3.60     0.0973   37.03 2.49e-25
#> speciesO. niveus   -10.07     0.7353  -13.69 6.27e-14
```

There are a few things to notice about this result. First, the object is a numeric matrix. This data structure was mostly likely chosen since all of the calculated results are numeric and a matrix object is stored more efficiently than a data frame. This choice was probably made in the late 1970's when the level of computational efficiency was critical. Second, the non-numeric data (the labels for the coefficients) are contained in the row names. Keeping the parameter labels as row names is very consistent with the conventions in the original S language. 

A reasonable course of action would be to create a visualization of the parameters values (perhaps using one of the plotting methods shown above). To do this, it would be sensible to convert the parameter matrix to a data frame. In doing so, a new column could be created with the variable names so that they can be used in the plot. However, note that several of the matrix column names would not be valid R object names (e.g. `"Pr(>|t|)"`.  Another complication is the consistency of the column names. For `lm` objects, the column for the test statistic is `"Pr(>|t|)"`. However, for other models, a different test might be used and, as a result, the column name is different (e.g., `"Pr(>|z|)"`) and the type of test is _encoded in the column name_.  
 
While these additional data formatting steps are not problematic they are a bit of an inconvenience, especially since they might be different for different types of models. The matrix is not a highly reusable data structure mostly because it must constrains the data to be of a single type (e.g. numeric). Additionally, keeping some data in the dimension names is also problematic since those data must be extracted to be of general use. For these reasons, the tidyverse places a large degree of importance on data frames and _tibbles_. Tibbles are data frames with a few extra features and, while they can use them, row names are eschewed. 

As a solution, the `broom` package has methods to convert many types of objects to a tidy structure. For example, using the `tidy()` method on the linear model produces:




```r
library(tidymodels)  # includes the broom package
tidy(main_effect_fit)
#> # A tibble: 3 x 5
#>   term             estimate std.error statistic  p.value
#>   <chr>               <dbl>     <dbl>     <dbl>    <dbl>
#> 1 (Intercept)         -7.21    2.55       -2.83 8.58e- 3
#> 2 temp                 3.60    0.0973     37.0  2.49e-25
#> 3 speciesO. niveus   -10.1     0.735     -13.7  6.27e-14
```
 
The column names are standardized across models and do not contain any additional data (such as the type of statistical test). The data previously contained in the row names are now in a column called `terms` and so on. One additional principle in the tidymodels ecosystem is that a functions return values should be **predictable, consistent, and unsurprising**. 
 
As another example of _unpredictability_, another convention in base R is related to missing data. The general rule is that missing data propagate more missing data; the average of a set of values with a missing data point is itself missing and so on. When models make predictions, the vast majority require all of the predictors to have complete values. There are several options based in to R at this point in the form of `na.action`.  This sets the policy for how a function should behave if there are missing values. The two most common policies are `na.fail` and `na.omit`. For former produces an error of missing data are involved while the latter removes the missing data prior to the calculations. From out previous example:


```r
# Add a missing value to the prediction set
new_values$temp[1] <- NA

# The predict method for `lm` defaults to `na.pass`:
predict(main_effect_fit, new_values)
#>    1    2    3    4    5    6 
#>   NA 50.4 54.0 57.6 61.2 64.8

# Alternatively 
predict(main_effect_fit, new_values, na.action = na.fail)
#> Error in na.fail.default(structure(list(temp = c(NA, 16L, 17L, 18L, 19L, : missing values in object

predict(main_effect_fit, new_values, na.action = na.omit)
#>    2    3    4    5    6 
#> 50.4 54.0 57.6 61.2 64.8
```

From a user's point of view, `na.omit()` can be problematic. In our example, `new_values` has 6 rows but only 5 would be returned. To compensate for this, the user would have to determine which row had the missing value and interleave a missing values in the appropriate place if the predictions were merged into `new_values`^[A base R policy called `na.exclude()` does exactly this.]. While it is rare that a prediction function uses `na.omit()` as its missing data policy, this does occur. Users who have determined this as the cause of an error in their code find it _quite memorable_. 

Finally, one other potential stumbling block can be inconsistencies between packages. Suppose a modeling project had an outcome with two classes. There are a variety of statistical and machine learning models that can be used. In order to produce class probability estimate for each sample, it is common for a model function to have a corresponding `predict()`method. However, there is significant heterogeneity in the argument values used by those methods to make class probability predictions. A sampling of these argument values for different models is: 

| Function     | Package      | Code                                       |
| :----------- | :----------- | :----------------------------------------- |
| `lda`        | `MASS`       | `predict(obj)`                             |
| `glm`        | `stats`      | `predict(obj, type = "response")`          |
| `gbm`        | `gbm`        | `predict(obj, type = "response", n.trees)` |
| `mda`        | `mda`        | `predict(obj, type = "posterior")`         |
| `rpart`      | `rpart`      | `predict(obj, type = "prob")`              |
| various      | `RWeka`      | `predict(obj, type = "probability")`       |
| `logitboost` | `LogitBoost` | `predict(obj, type = "raw", nIter)`        |
| `pamr.train` | `pamr`       | `pamr.predict(obj, type = "posterior")`    |

Note that the last example has a custom _function_ to make predictions instead of using the model common `predict()` interface.  

There are a few R packages that provide a unified interface to harmonize these modeling APIs, such as `caret` and `mlr`. tidymodels takes a similar approach to unification of the function interface as well as enforcing consistency in the function names and return values (e.g., `broom::tidy()`)^[If you've never seen `::` in R code before, it is a method to be explicit about what function you are calling. The value of the right-hand side is the _namespace_ where the function lives (usually a package name). The left-hand side is the function name. In cases where two packages use the same function name, this syntax will ensure that the correct function is invoked.].  

To resolve these usage issues, the tidymodels package have a few additional design goals that complement those of the tidyverse.


## Some additional tidy principals for modeling. 

### Be predictable, consistent, and unsurprising

Smooth out diverse interfaces etc. 

### Encourage empirical validation and good methodology.

Enable a wider variety of methodologies

Protect users from making objectively poor choices. Examples:

- *Information leakage* of training set data into evaluation sets.
- Analyzing integers as categories
- Down-sampling the test set



### Separate the user-interface from computational-interface


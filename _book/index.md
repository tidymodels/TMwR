
---
knit: "bookdown::render_book"
title: "Tidy Modeling with R"
author: ["Max Kuhn"]
date: "2019-12-12"
description: "Modeling of data is integral to science, business, politics, and many other aspects of our lives. The goals of this book are to: introduce neophytes to models and the tidyverse, demonstrate the `tidymodels` packages, and to outline good practices for the phases of the modeling process."
github-repo: topepo/TMwR
twitter-handle: topepos
documentclass: book
bibliography: [TMwR.bib]
biblio-style: apalike
link-citations: yes
colorlinks: yes
---

# Hello World {-} 

This is the website for _Tidy Modeling with R_. Its purpose is to be a guide to using a new collection of software in the R programming language that enable model building. There are few goals, depending on your background. First, if you are new to modeling and R, we hope to provide an introduction on how to use software to create models. The focus will be on a dialect of R called _the tidyverse_ that is designed to be a better interface for common tasks using R. If you've never heard of the tidyverse, there is a chapter that provides a solid introduction. The second (and primary) goal is to demonstrate how the tidyverse can be used to produce high quality models. The tools used to do this are referred to as the _tidymodels packages_. The third goal is to use the tidymodels packages to encourage good methodology and statistical practice. Many models, especially more complex predictive or machine learning models, can be created to work very well on the data at hand but may fail when exposed to new data. Often, this issue is due to poor choices that were made during the development and/or selection of the models. Whenever possible, our software attempts to prevent this from occurring but common pitfalls are discussed in the course of describing and demonstrating the software. 

This book is not intended to be a reference on different types of models. We suggest other resources to learn the nuances of models. A general source for information about the most common type of model, the _linear model_, we suggest @fox08. Another excellent resource for investigating and analyzing data is @wickham2016. For predictive models, @apm is a good resource. For pure machine learning methods, @Goodfellow is an excellent (but formal) source of information.  In some cases, we describe some models that are used in this text but in a way that is less mathematical (and hopefully more intuitive). 

We do not assume that readers will have had extensive experience in model building and statistics. Some statistical knowledge is required, such as: random sampling, variance, correlation, basic linear regression, and other topics that are usually found in a basic undergraduate statistics or data analysis course. 

This website is __free to use__, and is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivs 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/us/) License. The sources used to create the book can be found at [`github.com/topepo/TMwR`](https://github.com/topepo/TMwR). We use the [`bookdown`](https://bookdown.org/) package to create the website [@bookdown]. One reason that we chose this license and this technology for the book is so that we can make it _completely reproducible_; all of the code and data used to create it are free and publicly available. 

_Tidy Modeling with R_ is currently a work in progress. As we create it, this website is updated. Be aware that, until it is finalized, the content and/or structure of the book may change. 

This openness also allows users to contribute if they wish. Most often, this comes in the form of correcting types, grammar, and other aspects of our work that could use improvement. Instructions for making contributions can be found in the [`contributing.md`](https://github.com/topepo/TMwR/blob/master/contributing.md) file. Also, be aware that this effort has a code of conduct, which can be found at [`code_of_conduct.md`](https://github.com/topepo/TMwR/blob/master/code_of_conduct.md). 

In terms of software lifecycle, the tidymodels packages are fairly young. We will do our best to maintain backwards compatibility and, at the completion of this work, will archive the specific versions of software that were used to produce it. The primary packages, and their versions, used to create this website are:




```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.1 (2019-07-05)
#>  os       macOS Mojave 10.14.6        
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/New_York            
#>  date     2019-12-12                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package     * version      date       lib source                           
#>  AmesHousing * 0.0.3        2017-12-17 [1] CRAN (R 3.6.0)                   
#>  bookdown    * 0.14         2019-10-01 [1] CRAN (R 3.6.0)                   
#>  broom       * 0.5.2        2019-04-07 [1] CRAN (R 3.6.0)                   
#>  dials       * 0.0.4        2019-12-02 [1] CRAN (R 3.6.1)                   
#>  discrim     * 0.0.1        2019-10-10 [1] local                            
#>  dplyr       * 0.8.3        2019-07-04 [1] CRAN (R 3.6.0)                   
#>  ggplot2     * 3.2.1.9000   2019-12-06 [1] local                            
#>  infer       * 0.5.1        2019-11-19 [1] CRAN (R 3.6.0)                   
#>  parsnip     * 0.0.4.9000   2019-12-04 [1] local                            
#>  purrr       * 0.3.3        2019-10-18 [1] CRAN (R 3.6.0)                   
#>  recipes     * 0.1.7.9002   2019-12-10 [1] local                            
#>  rlang         0.4.2.9000   2019-12-01 [1] Github (r-lib/rlang@1be25e7)     
#>  rsample     * 0.0.5        2019-07-12 [1] CRAN (R 3.6.0)                   
#>  tibble      * 2.99.99.9010 2019-12-06 [1] Github (tidyverse/tibble@f4365f7)
#>  tune        * 0.0.0.9004   2019-12-09 [1] local                            
#>  workflows   * 0.0.0.9002   2019-11-29 [1] local                            
#>  yardstick   * 0.0.4        2019-08-26 [1] CRAN (R 3.6.0)                   
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```

[`pandoc`](https://pandoc.org/) is also instrumental in creating this work. The version used here is 2.3.1. 

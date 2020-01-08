
---
knit: "bookdown::render_book"
title: "Tidy Modeling with R"
author: ["Max Kuhn"]
date: "2020-01-08"
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

This is the website for _Tidy Modeling with R_. Its purpose is to be a guide to using a new collection of software in the R programming language that enable model building. There are few goals, depending on your background. First, if you are new to modeling and R, we hope to provide an introduction on how to use our software to create models. The focus will be on a dialect of R called _the tidyverse_ that is designed to be a better interface for common tasks using R. If you've never heard of the tidyverse, there is a chapter that provides a solid introduction. The second (and primary) goal is to demonstrate how the tidyverse can be used to produce high quality models. The tools used to do this are referred to as the _tidymodels packages_. The third goal is to use the tidymodels packages to encourage good methodology and statistical practice. Many models, especially complex predictive or machine learning models, can work very well on the data at hand but may also fail when exposed to new data. Often, this issue is due to poor choices that were made during the development and/or selection of the models. Whenever possible, our software attempts to prevent these and other pitfalls. 

This book is not intended to be a reference on different types of these techniques We suggest other resources to learn the nuances of models. A general source for information about the most common type of model, the _linear model_, we suggest @fox08. Another excellent resource for investigating and analyzing data is @wickham2016. For predictive models, @apm is a good resource. For pure machine learning methods, @Goodfellow is an excellent (but formal) source of information.  In some cases, we describe some models that are used in this text but in a way that is less mathematical (and hopefully more intuitive). 

We do not assume that readers will have had extensive experience in model building and statistics. Some statistical knowledge is required, such as: random sampling, variance, correlation, basic linear regression, and other topics that are usually found in a basic undergraduate statistics or data analysis course. 

_Tidy Modeling with R_ is currently a work in progress. As we create it, this website is updated. Be aware that, until it is finalized, the content and/or structure of the book may change. 

This openness also allows users to contribute if they wish. Most often, this comes in the form of correcting typos, grammar, and other aspects of our work that could use improvement. Instructions for making contributions can be found in the [`contributing.md`](https://github.com/topepo/TMwR/blob/master/contributing.md) file. Also, be aware that this effort has a code of conduct, which can be found at [`code_of_conduct.md`](https://github.com/topepo/TMwR/blob/master/code_of_conduct.md). 

In terms of software lifecycle, the tidymodels packages are fairly young. We will do our best to maintain backwards compatibility and, at the completion of this work, will archive the specific versions of software that were used to produce it. The primary packages, and their versions, used to create this website are:




```
#> ─ Session info ───────────────────────────────────────────────────────────────
#>  setting  value                       
#>  version  R version 3.6.1 (2019-07-05)
#>  os       macOS Catalina 10.15.1      
#>  system   x86_64, darwin15.6.0        
#>  ui       X11                         
#>  language (EN)                        
#>  collate  en_US.UTF-8                 
#>  ctype    en_US.UTF-8                 
#>  tz       America/New_York            
#>  date     2020-01-08                  
#> 
#> ─ Packages ───────────────────────────────────────────────────────────────────
#>  package     * version    date       lib source                      
#>  AmesHousing * 0.0.3      2017-12-17 [1] CRAN (R 3.6.0)              
#>  bookdown    * 0.16       2019-11-22 [1] CRAN (R 3.6.0)              
#>  broom         0.5.3      2019-12-14 [1] CRAN (R 3.6.0)              
#>  dplyr       * 0.8.3      2019-07-04 [1] CRAN (R 3.6.0)              
#>  ggplot2     * 3.2.1      2019-08-10 [1] CRAN (R 3.6.0)              
#>  purrr       * 0.3.3      2019-10-18 [1] CRAN (R 3.6.0)              
#>  rlang         0.4.2.9000 2019-12-26 [1] Github (r-lib/rlang@ce4f717)
#>  tibble      * 2.1.3      2019-06-06 [1] CRAN (R 3.6.0)              
#> 
#> [1] /Library/Frameworks/R.framework/Versions/3.6/Resources/library
```

[`pandoc`](https://pandoc.org/) is also instrumental in creating this work. The version used here is 2.3.1. 

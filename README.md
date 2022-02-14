# TMwR

[![Build Status](https://github.com/tidymodels/TMwR/workflows/bookdown/badge.svg)](https://github.com/tidymodels/TMwR/actions)




This repository contains the source for [_Tidy Modeling with R_](https://tmwr.org). The purpose of this book is to demonstrate how the [tidyverse](https://www.tidyverse.org/) and [tidymodels](https://www.tidymodels.org/) can be used to produce high quality models.

# Reproducing the book or results

First, you'll need to install the required packages. To do this, first install the `remotes` package:

``` r
install.packages("remotes")
```

Then use this to install what you need to create the book: 

``` r
remotes::install_github("tidymodels/TMwR")
```

Although we rigorously try to use the current CRAN versions of all packages, the code above may install some development versions. 

The content is created using the `bookdown` package. To compile the book, use:

```r
bookdown::render_book("index.Rmd", "bookdown::gitbook")
```

This will create the HTML files in a directory called `_book`. Although we are in the process of publishing a print version of this work with O'Reilly, we do _not_ currently support building to a PDF version.


# Contributing

Please note that this work is written under a [Contributor Code of Conduct](CODE_OF_CONDUCT.md) and the online version is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). By participating in this project (for example, by submitting an [issue](https://github.com/tidymodels/TMwR/issues) with suggestions or edits) you agree to abide by its terms. Instructions for making contributions can be found in the [`contributing.md`](contributing.md) file.

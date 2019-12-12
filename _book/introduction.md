


# Introduction

Models are mathematical tools that create equations that are intended to mimic the data given to them. These equations can be used for various purposes, such as: predicting future events, determining if there is a different between two groups, as an aid to a map-based visualization, discovering novel patterns in the data that could be further investigated, and so on. Their utility hinges on their ability to be reductive; the primary influences in the data can be captured mathematically in a way that is useful. 

In the last 20 years, mathematical models have become ubiquitous in our daily lives, in both obvious and subtle ways. A typical day for many people might involve checking the weather to see when a good time would be to walk the dog, ordering a product from a website, typing (and autocorrecting) a text message to a friend, and checking email. In each of these instances, there is a good chance that some type of model was used in an assistive way. In some cases, the contribution of the model might be easily perceived ("You might also be interested in purchasing product _X_") while in other cases the impact was the absence of something (e.g., spam email). Models are used to choose clothing that a customer might like, a molecule that should be evaluated as a drug candidate, and might even be the mechanism that a nefarious company uses avoid the discovery of cars that over-pollute.  For better or worse, models are here to stay.

Two reasons that models permeate our lives are that software exists that facilitates their creation and that data has become more easily captured and accessible. In regard to software, it is obviously critical that software produces the _correct_ equations that represent the data. For the most part, determining mathematical correctness is possible. However, the creation of an appropriate model hinges on a few other aspects. 

First, it is important that it is easy to operate the software in a _proper way_. For example, the user interface should not be so arcane that the user would not know that they have inappropriately specified the wrong information. As an analogy, one might have a high quality kitchen measuring cup capable of great precision but if the chef adds a cup of salt instead of a cup of sugar, the results would be unpalatable. As a specific example of this issue, @baggerly2009 report myriad problems in the data analysis in a high profile computational biology publication. One of the issues was related to how the users were required to add the names of the model inputs. The user-interface of the software was poor enough that it was easy to _offset_ the column names of the data from the actual data columns. In the analysis of the data, this resulted in the wrong genes being identified as important for treating cancer patients. This, and many other issues, led to the stoppage of numerous clinical trials [@Carlson2012]. If we are to expect high quality models, it is important that the software facilitate proper usage. @abrams2003 describes an interesting principle to live by: 

> The Pit of Success: in stark contrast to a summit, a peak, or a journey across a desert to find victory through many trials and surprises, we want our customers to simply fall into winning practices by using our platform and frameworks. 

Data analysis software should also espouse this idea. 

The second important aspect of model building is related to _scientific methodology_. For models that are used to make complex predictions, it can be easy to unknowingly commit errors related to logical fallacies or inappropriate assumptions. Many machine learning models are so adept at finding patterns, they can effortlessly find empirical patterns in the data that fail to reproduce later. Some of these types of methodological errors are insidious in that the issue might be undetectable until a later time when new data that contain the true result are obtained. In short, as our models become more powerful and complex it has also become easier to commit latent errors. This relates to software. Whenever possible, the software should be able to protect users from committing such mistakes. Here, software should make it easy for users to do the right thing. 

These two aspects of model creation are crucial. Since tools for creating models are easily obtained and models can have such a profound impact, many more people are creating them. In terms of technical expertise and training, their backgrounds will vary. It is important that their tools be _robust_ to the experience of the user. On one had, they tools should be powerful enough to create high-performance models but, one the other hand, should be easy to use in an appropriate way.  This book describes a suite of software that can can create different types of models. The software has been designed with these additional characteristics in mind.

The software is based on the R programming language [@baseR]. R has been designed especially for data analysis and modeling. It is based on the _S language_ which was created in the 1970's to

> "to turn ideas into software**, **quickly and faithfully" [@Chambers:1998]

R has a vast ecosystem of _packages_; these are mostly user-contributed modules that focus on a specific theme, such as modeling, visualization, and so on. 

One collection of packages is called the **_tidyverse_** [@tidyverse]. The tidyverse is an opinionated collection of R packages designed for data science. All packages share an underlying design philosophy, grammar, and data structures. Several of these design philosophies are directly related to the aspects of software described above. Within the tidyverse, there is a set of packages specifically focused on modeling and these are usually referred to as the ***tidymodels*** packages.  

## Let's talk about models


### Types of Models

descriptive, predictive, inferential

regression, classification

### Some terminology

supervise, unsupervised

types of predictors


### Where does modeling fit into the scientific process? 

(probably need to get explicit permission to use this)

<div class="figure" style="text-align: center">
<img src="figures-premade/data-science-model.svg" alt="The data science process (from _R for Data Science_)." width="80%" />
<p class="caption">(\#fig:data-science-model)The data science process (from _R for Data Science_).</p>
</div>


### Modeling is a _process_, not a single activity


(probably need to get explicit permission to use this too)


<div class="figure" style="text-align: center">
<img src="figures/introduction-modeling-process-1.svg" alt="A schematic for the typical modeling process." width="100%" />
<p class="caption">(\#fig:modeling-process)A schematic for the typical modeling process.</p>
</div>



## What does it mean to be "tidy"


## Outline of future chapters


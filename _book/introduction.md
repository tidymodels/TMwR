


# Introduction

Models are mathematical tools that create equations that are intended to mimic the data given to them. These equations can be used for various purposes, such as: predicting future events, determining if there is a difference between several groups, as an aid to a map-based visualization, discovering novel patterns in the data that could be further investigated, and so on. Their utility hinges on their ability to be reductive; the primary influences in the data can be captured mathematically in a way that is useful. 

Since the start of the 21st century, mathematical models have become ubiquitous in our daily lives, in both obvious and subtle ways. A typical day for many people might involve checking the weather to see when a good time would be to walk the dog, ordering a product from a website, typing (and autocorrecting) a text message to a friend, and checking email. In each of these instances, there is a good chance that some type of model was used in an assistive way. In some cases, the contribution of the model might be easily perceived ("You might also be interested in purchasing product _X_") while in other cases the impact was the absence of something (e.g., spam email). Models are used to choose clothing that a customer might like, a molecule that should be evaluated as a drug candidate, and might even be the mechanism that a nefarious company uses avoid the discovery of cars that over-pollute.  For better or worse, models are here to stay.

Two reasons that models permeate our lives are that software exists that facilitates their creation and that data has become more easily captured and accessible. In regard to software, it is obviously critical that software produces the _correct_ equations that represent the data. For the most part, determining mathematical correctness is possible. However, the creation of an appropriate model hinges on a few other aspects. 

First, it is important that it is easy to operate the software in a _proper way_. For example, the user interface should not be so arcane that the user would not know that they have inappropriately specified the wrong information. As an analogy, one might have a high quality kitchen measuring cup capable of great precision but if the chef adds a cup of salt instead of a cup of sugar, the results would be unpalatable. As a specific example of this issue, @baggerly2009 report myriad problems in the data analysis in a high profile computational biology publication. One of the issues was related to how the users were required to add the names of the model inputs. The user-interface of the software was poor enough that it was easy to _offset_ the column names of the data from the actual data columns. In the analysis of the data, this resulted in the wrong genes being identified as important for treating cancer patients. This, and many other issues, led to the stoppage of numerous clinical trials [@Carlson2012]. 

If we are to expect high quality models, it is important that the software facilitate proper usage. @abrams2003 describes an interesting principle to live by: 

> The Pit of Success: in stark contrast to a summit, a peak, or a journey across a desert to find victory through many trials and surprises, we want our customers to simply fall into winning practices by using our platform and frameworks. 

Data analysis software should also espouse this idea. 

The second important aspect of model building is related to _scientific methodology_. For models that are used to make complex predictions, it can be easy to unknowingly commit errors related to logical fallacies or inappropriate assumptions. Many machine learning models are so adept at finding patterns, they can effortlessly find empirical patterns in the data that fail to reproduce later. Some of these types of methodological errors are insidious in that the issue might be undetectable until a later time when new data that contain the true result are obtained. In short, as our models become more powerful and complex it has also become easier to commit latent errors. This also relates to programming. Whenever possible, the software should be able to protect users from committing such mistakes. Software should make it easy for users to do the right thing. 

These two aspects of model creation are crucial. Since tools for creating models are easily obtained and models can have such a profound impact, many more people are creating them. In terms of technical expertise and training, their backgrounds will vary. It is important that their tools be _robust_ to the experience of the user. On one had, they tools should be powerful enough to create high-performance models but, on the other hand, should be easy to use in an appropriate way.  This book describes a suite of software that can can create different types of models. The software has been designed with these additional characteristics in mind.

The software is based on the R programming language [@baseR]. R has been designed especially for data analysis and modeling. It is based on the _S language_ which was created in the 1970's to

> "turn ideas into software, quickly and faithfully" [@Chambers:1998]

R is open-source and is provided free of charge. It is a powerful programming language that can be used for many different purposes but specializes in data analysis, modeling, and machine learning. R is easily _extensible_; it has a vast ecosystem of *packages*; these are mostly user-contributed modules that focus on a specific theme, such as modeling, visualization, and so on.

One collection of packages is called the **_tidyverse_** [@tidyverse]. The tidyverse is an opinionated collection of R packages designed for data science. All packages share an underlying design philosophy, grammar, and data structures. Several of these design philosophies are directly related to the aspects of software described above. If you've never used the tidy verse packages, Chapter \@ref(tidyverse-primer) contains a review. Within the tidyverse, there is a set of packages specifically focused on modeling and these are usually referred to as the ***tidymodels*** packages. This book is an extended software manual for conducting modeling using the tidyverse. It shows how to use a set of packages, each with its own specific purpose, together to create high-quality models.  

## Types of models

Before proceeding, lets describe a taxa for types of models, grouped by _purpose_. While not exhaustive,  most models fail into _at least_ one of these categories: 

**Descriptive Models**: The purpose here would be to model the data so that it can be used to describe or illustrate characteristics of some data.  The analysis might have no other purpose than to visually emphasize some trend or artifact in the data. 

For example, large scale measurements of RNA have been possible for some time using _microarrays_. Early laboratory methods placed a biological sample on a small microchip. Very small locations on the chip would be able to assess a measure of signal based on the abundance of a specific RNA sequence. The chip would contain thousands (or more) outcomes, each a quantification of the RNA related to some biological process. However, there could be quality issued on the chip that might lead to poor results. A fingerprint accidentally left on a portion of the chip might cause inaccurate measurements when scanned. 

An early methods for evaluating such issues where _probe-level models_, or PLM's [@bolstad2004]. A statistical model would be created that accounted for the _known_ differences for the data from the chip, such as the RNA sequence, the type of sequence and so on. If there were other, unwanted factors in the data, these would be contained in the model residuals. When the residuals were plotted by their location on the chip, a good quality chip would show no patterns. When an issue did occur, some sort of spatial pattern would be discernible. Often the type of pattern would suggest the underlying issue (e.g. a fingerprint) and a possible solution (wipe the chip off and rescan). Figure \@ref(fig:descr-examples)(a) shows an application of this method for two microarrays taken from @Gentleman2005. The images show two different colors; red is where the signal intensity was larger than the model expects while the blue color shows lower than expected values. The left-hand panel demonstrates a fairly random pattern while the right-hand panel shows some type of unwanted artifact. 

<div class="figure" style="text-align: center">
<img src="figures/introduction-descr-examples-1.png" alt="Two examples of how descriptive models can be used to illustrate specific patterns." width="80%" />
<p class="caption">(\#fig:descr-examples)Two examples of how descriptive models can be used to illustrate specific patterns.</p>
</div>

Another example of a descriptive model is the _locally estimated scatterplot smoothing_ model, more commonly known as LOESS [@cleveland1979]. Here, a smooth and flexible regression model is fit to a data set, usually with a single independent variable, and the fitted regression line is used to elucidate some trend in the data. These types of _smoothers_ are used to discover potential ways to represent a variable in a model. This is demonstrated in Figure \@ref(fig:descr-examples)(b) where a nonlinear trend is illuminated by the flexible smoother. 


**Inferential Models**: In these situations, the goal is to produce a decision for a research question or to test a specific hypothesis. The goal is to make some statement of truth regarding some predefined conjecture or idea. In many (but not all) cases, some qualitative statement is produced.

For example, in a clinical trial, the goal might be to provide confirmation that a new therapy does a better job in prolonging life than an alternative (e.g., an existing therapy or no treatment). If the clinical endpoint was related to survival or a patient, the _null hypothesis_ might be that the two therapeutic groups have equal median survival times with the alternative hypothesis being that the new therapy has higher median survival.  If this trial were evaluated using the traditional *null hypothesis significance testing* (NHST), a p-value would be produced using some pre-defined methodology based on a set of assumptions for the data. Small values of the p-value indicate that there is evidence that the new therapy does help patients live longer. If not, the conclusion is that there is a failure to show such an difference (which could be due to a number of reasons). 

What are the important aspects of this type of analysis? Inferential techniques typically produce some type of probabilistic output, such as a p-value, confidence interval, or posterior probability. Generally, to compute such a quantity, formal assumptions must be made about the data and the underlying processes that generated the data. The quality of the statistical results are highly dependent on these pre-defined  assumptions as well as how much the observed data appear to agree with them. The most critical factors here are theoretical in nature: if my data were independent and follow distribution _X_, then test statistic _Y_ can be used to produce a p-value. Otherwise, the resulting p-value might be inaccurate.

One aspect of inferential analyses is that there _tends_ to be a longer feedback loop that could help understand how well the data fit the assumptions. In our clinical trial example, if statistical (and clinical) significance indicated that the new therapy should be available for patients to use, it may be years before it is used in the field and enough data were generated to have an independent assessment of whether the original statistical analysis led to the appropriate decision. 

**Predictive Models**: There are occasions where data are modeled in an effort to produce the most accurate prediction possible for new data. Here, the primary goal is that the predicted values have the highest possible fidelity to the true value of the new data. 

A simple example would be for a book buyer to predict how many copies of a particular book should be shipped to his/her store for the next month. An over-prediction wastes space and money due to excess books. If the prediction is smaller than it should be, there is opportunity loss and less profit. 

For this type of model, the problem type is one of _estimation_ rather than inference. For example, the buyer is usually not concerned with a question such as "Will I sell more than 100 copies of book _X_ next month?" but rather "How many copies of _X_ will customers purchase next month?" Also, depending on the context, there may not be any interest in _why_ the predicted value is _X_. In other words, is more interest in the value itself than evaluating a formal hypothesis related to the data. That said, the prediction can also include measures of uncertainty. In the case of the book buyer, some sort of forecasting error might be valuable to help them decide on how many to purchase or could serve as a metric to gauge how well the prediction method worked.  

What are the most important factors affecting predictive models? There are many different ways that a predictive model could be created. The important factors depend on how the model was developed.

For example, a _mechanistic model_ could be developed based on first principles to produce a model equation that is dependent on assumptions. For example, when predicting the amount of a drug that is in a person's body at a certain time, some formal assumptions are made on how the drug is administered, absorbed, metabolized, and eliminated. Based on this, a set of differential equations can be used to derive a specific model equation. Data are used to estimate the known parameters of this equation and predictions are made after parameter estimation. Like inferential models,  mechanistic predictive models greatly depend on the assumptions that define their model equations. However, unlike inferential models, it is easy to make data-driven statements about how well the model performs based on how well it predicts the existing data. Here the feedback loop for the modeler is much faster than it would be for a hypothesis test. 

_Empirically driven models_ are those that have more vague assumptions that are used to create their model equations. These models tend to fall more into the machine learning category. A good example is the simple _K_-nearest neighbor (KNN) model. Given a set of reference data, a new sample is predicted by using the values of the most similar data in the reference set. For example, if a book buyer needs a prediction for a new book, historical data from existing books may be available. A 5-nearest neighbor model would estimate the amount of the new book to purchase based on the sales numbers of the five books that are most similar to the new one (for some definition of "similar"). This model is only defined by the structure of the prediction (the average of five similar books). No theoretical or probabilistic assumptions are made about the sales numbers or the variables that are used to define similarity. In fact, the primary method of evaluating the appropriateness of the model is to assess its accuracy using existing data. If the structure of this type of model was a good choice, the predictions would not be close to the actual values. 

Broader discussions of these distinctions can be found in @breiman2001 and @shmueli2010. Note that we have defined the type of model by how it is used rather than its mathematical qualities. An ordinary linear regression model might fall into all three classes of models, depending on how it is used: 

* Descriptive smoother, similar to LOESS, called _restricted smoothing splines_ [@Durrleman1989] can be used to describe trends in data using ordinary linear regression with specialized terms. 

* An _analysis of variance_ (ANOVA) model is a popular method for producing the p-values used for inference. ANOVA models are a special case of linear regression. 

* If a simple linear regression model produces highly accurate predictions, it can be used as a predictive model. 

However, there are many more examples of predictive models that cannot (or at least should not) be used for inference. Even if probabilistic assumptions were made for the data, the nature of the KNN model makes the math required for inference intractable. 

There is an additional connection between the types of models. While the primary purpose of descriptive and inferential models might not be related to prediction, the predictive capacity of the model should not be ignored. For example, logistic regression is a popular model for data where the outcome is qualitative with two possible values. It can model how variables related to the probability of the outcomes. When used in an inferential manner, there is usually an abundance of attention paid to the _statistical qualities_ of the model. For example, analysts tend to strongly focus on the selection of which independent variables are contained in the model. Many iterations of model building are usually used to determine a minimal subset of independent variables that have a  "statistically significant" relationship to the outcome variable. This is usually achieved when all of the p-values for the independent variables are below some value (e.g. 0.05). From here, the analyst typically focuses on making qualitative statements about the relative influence that the variables have on the outcome.  

A potential problem with this approach is that it can be dangerous when statistical significance is used as the _only_ measure of model quality.  It is certainly possible that this statistically optimized model has poor model accuracy (or some other measure of predictive capacity). While the model might not be used for prediction, how much should the inferences be trusted from a model that has all significant p-values but a  dismal accuracy? Predictive performance tends to be related to how close the model's fitted values are to the observed data. If the model has limited fidelity to the data, the inferences generated by the model should be highly suspect. In other words, statistical significance may not imply that the model should be used. This may seem intuitively obvious, but is often ignored in real-world data analysis.

## Some terminology

Before proceeding, some additional terminology related to modeling, data, and other quantities should be outlined. These descriptions are not exhaustive. 

First, many models can be categorized as being _supervised_ or _unsupervised_. Unsupervised models are those that seek patterns, clusters, or other characteristics of the data but lack an outcome variable (i.e., a dependent variable). For example, principal component analysis (PCA), clustering, and autoencoders are used to understand relationships between variables or sets of variables without an explicit relationship between variables and an outcome. Supervised models are those that have an outcome variable. Linear regression, neural networks, and numerous other methodologies fall into this category. Within supervised models, the two main sub-categories are: 

 * _Regression_, where a numerical outcome is being predicted.

 * _Classification_, where the outcome is an ordered or unordered set of _qualitative_ values.  

These are imperfect definitions and do not account for all possible types of models. In coming chapters, we refer to these types of supervised techniques as the _model mode_. 

In terms of data, the main species are quantitative and qualitative. Examples of the former are real numbers and integers. Qualitative values, also known as nominal data, are those that represent some sort of discrete state that cannot be placed on a numeric scale. 

Different variables can have different _roles_ in an analysis. Outcomes (otherwise known as the labels, endpoints, or dependent variables) are the value being predicted in supervised models. The independent variables, which are the substrate for making predictions of the outcome, also referred to as predictors, features, or covariates (depending on the context). Here, the terms _outcomes_ and _predictors_ are used most frequently here. 

## How does modeling fit into the data analysis/scientific process? {#model-phases}

In what circumstances are model created? Are there steps that precede such an undertaking? Is it the first step in data analysis? 

There are always a few critical phases of data analysis that come before modeling. First, there is the chronically underestimated process of **cleaning the data**. No matter the circumstances, the data should be investigated to make sure that it is well understood, applicable to the project goals, accurate, and appropriate. These steps can easily take more time than the rest of the data analysis process (depending on the circumstances). 

Data cleaning can also overlap with the second phase of **understanding the data**, often referred to as exploratory data analysis (EDA). There should be knowledge of how the different variables related to one another, their distributions, typical ranges, and other attributes. A good question to ask at this phase is "how did I come by _these_ data?" This question can help understand how the data at-hand have been sampled or filtered and if these operations were appropriate. For example, when merging data base tables, a join may go awry that could accidentally eliminate one or more sub-populations of samples. Another good idea would be to ask if the data are _relavant_. For example, to predict whether patients have Alzheimer's disease or not, it would be unwise to have a data set containing subject with the disease and a random sample of healthy adults from the general population. Given the progressive nature of the disease, the model my simply predict who the are the _oldest patients_. 

Finally, before starting a data analysis process, there should be clear expectations of the goal of the model and how performance (and success) will be judged. At least one _performance metric_ should be identified with realistic goals of what can be achieved. Common statistical metrics are classification accuracy, true and false positive rates, root mean squared error, and so on. The relative benefits and drawbacks of these metrics should be weighted. It is also important that the metric be germane (i.e., alignment with the broader data analysis goals is critical). 

<div class="figure" style="text-align: center">
<img src="figures-premade/data-science-model.svg" alt="The data science process (from _R for Data Science_)." width="80%" />
<p class="caption">(\#fig:data-science-model)The data science process (from _R for Data Science_).</p>
</div>

The process of investigating the data may not be simple. @wickham2016 contains an excellent illustration of the general data analysis process, reproduced with Figure \@ref(fig:data-science-model). Data ingestion and cleaning are shown as the initial steps. When the analytical steps commence, they are a heuristic process; we cannot pre-determine how long they may take. The cycle of analysis, modeling, and visualization often require multiple iterations. 

<div class="figure" style="text-align: center">
<img src="figures/introduction-modeling-process-1.svg" alt="A schematic for the typical modeling process (from _Feature Engineering and Selection_)." width="100%" />
<p class="caption">(\#fig:modeling-process)A schematic for the typical modeling process (from _Feature Engineering and Selection_).</p>
</div>

This iterative process is especially true for modeling. Figure \@ref(fig:modeling-process) originates from @kuhn20202 and is meant to emulate the typical path to determining an appropriate model. The general phases are:

 * Exploratory data analysis (EDA) and Quantitative Analysis (blue bars). Initially there is a back and forth between numerical analysis and visualization of the data (represented in Figure \@ref(fig:data-science-model)) where different discoveries lead to more questions and data analysis "side-quests" to gain more understanding. 
 * Feature engineering (green bars). This understanding translated to the creation of specific model terms that make it easier to accurately model the observed data. This can include complex methodologies (e.g., PCA) or simpler features (using the ratio of two predictors). 

 * Model tuning and selection (red and gray bars). A variety of models are generated and their performance is compared. Some models require _parameter tuning_ where some structural parameters are required to be specified or optimized. 

After an initial sequence of these tasks, more understanding is gained regarding which types of models are superior as well as which sub-populations of the data are not being effectively estimated. This leads to additional EDA and feature engineering, another round of modeling, and so on. Once the data analysis goals are achieved, the last steps are typically to finalize and document the model. For predictive models, it is common at the end to validate the model on an additional set of data reserved for this specific purpose. 


## Where does the model begin and end? {#begin-model-end}

So far, we have defined the model to be a structural equation that relates some predictors to one or more outcomes. Let's consider ordinary linear regression as a simple and well known example. The outcome data are denoted as $y_i$, where there are $i = 1 \ldots n$ samples in the data set. Suppose that there are $p$ predictors $x_{i1}, \ldots, x_{ip}$ that are used to predict the outcome. Linear regression produces a model equation of 

$$ \hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_{i1} + \ldots + \hat{\beta}_px_{ip} $$

While this is a _linear_ model, it is only linear in the parameters. The predictors could be nonlinear terms (such as the $log(x_i)$). 

The conventional way of thinking is that the modeling _process_ is encapsulated by the model. For many data sets that are straight-forward in nature, this is the case. However, there are a variety of _choices_ and additional steps that often occur before the data are ready to be added to the model. Some examples:

* While our model has $p$ predictors, it is common to start with more than this number of candidate predictors. Through exploratory data analysis or previous experience, some of the predictors may be excluded from the analysis. In other cases, some feature selection algorithm may have been used to make a data-driven choice for the minimum predictors set to be used in the model. 
* There are times when the value of an important predictor is know known. Rather than eliminating this value from the data set, it could be _imputed_ using other values in the data. For example, if $x_1$ were missing but was correlated with predictors $x_2$ and $x_3$, an imputation method could estimate the missing $x_1$ observation from the values of $x_2$ and $x_3$. 
* As previously mentioned, it may be beneficial to transform the scale of a predictor. If there is **not** _a priori_ information on what the new scale should be, it might be estimated using a transformation technique. Here, the existing data would be used to statistically _estimate_ the proper scale that optimizes some criterion. Other transformations, such as the previously mentioned PCA, take groups of predictors and transform them into new features that are used as the predictors.

While the examples above are related to steps that occur before the model, there may also be operations that occur after the model is created. For example, when a classification model is created where the outcome is binary (e.g., `event` and `non-event`), it is customary to use a 50% probability cutoff to create a discrete class prediction (also known as a "hard prediction"). For example, a classification model might estimate that the probability of an event was 62%. Using the typical default, the hard prediction would be `event`. However, the model may need to be more focused on reducing false positive results (i.e., where true non-events are classified as events). One way to do this is to _raise_ the cutoff from 50% to some greater value. This increases the level of evidence required to call a new sample as an event. While this reduces the true positive rate (which is bad), it may have a more profound effect on reducing false positives. The choice of the cutoff value should be optimized using data. This is an example of a post-processing step that has a significant effect on how well the model works even though it is not contained in the model fitting step. 

These examples have a common characteristic of requiring data for derivations that alter the raw data values or the predictions generated by the model. 

It is very important to focus on the broader _model fitting process_ instead of the specific model being used to estimate parameters. This would include any pre-processing steps, the model fit itself, as well as potential post-processing activities. In this text, this will be referred to as the **model workflow** and would include any data-driven activities that are used to produce a final model equation. 

This will come into play when topics such as resampling (Chapter \@ref(resampling)) and  model tuning are discussed. Chapter \@ref(workflows) describes software for creating a model workflow. 


## Outline of future chapters

The first order of business is to introduce (or review) the ideas and syntax of the tidyverse in Chapter \@ref(tidyverse-primer). In this chapter, we also summarize the unmet needs for modeling when using R. This provides good motivation for why model-specific tidyverse techniques are needed. This chapter also outlines some additional principles related to this challenges. 

Chapter \@ref(two-models) shows two different data analyses for the same data where one is focused on prediction and the other is for inference. This should illustrates the challenges for each approach and what issues are most relavant for each.  


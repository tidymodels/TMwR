

# Ensembles of Models {#ensembles}


A model ensemble, where the predictions of multiple single learners are aggregated together to make one prediction, can produce a high-performance final model. The most popular methods for creating ensemble models are bagging [@breiman1996bagging], random forest [@ho1995random; @breiman2001random], and boosting [@freund1997decision]. Each of these methods combines the predictions from multiple versions of the same type of model (e.g., classifications trees). However, one of the earliest methods for creating ensembles is *model stacking* [@wolpert1992stacked; @breiman1996stacked]. 

:::rmdnote
Model stacking combines the predictions for multiple models of any type. For example, a logistic regression, classification tree, and support vector machine can be included in a stacking ensemble. 
:::


This chapter shows how to stack predictive models using the <span class="pkg">stacks</span> package. We'll re-use the results from Chapter \@ref(workflow-sets) where multiple models were evaluated to predict the compressive strength of concrete mixtures.

The process of building a stacked ensemble is:

1.  Assemble the training set of hold-out predictions (produced via resampling).
2.  Create a model to blend these predictions.
3.  For each member of the ensemble, fit the model on the original training set.

In subsequent sections, we'll describe this process. However, before proceeding, there is some nomenclature to clarify around the different variations of what we can mean by "the model". This can quickly become an overloaded term when we are working on a complex modeling analysis! Let's consider the multilayer perceptron model (MLP, a.k.a. neural network) created in Chapter \@ref(workflow-sets).

In general, we'll talk about a "multilayer perceptron model" as the *type* of model. Linear regression and support vector machines are other model types.

One important aspect of a model are its tuning parameters. Back in Chapter \@ref(workflow-sets), the MLP model was tuned over 25 tuning parameter values. In the previous chapters, we've called these *candidate tuning parameter* values or *model configurations*. In literature on ensembling these have also been called the "base models". 

:::rmdnote
We'll use the term "candidate members" to describe the possible model configurations (of all model types) that might be included in the stacking ensemble.
:::

This means that a stacking model can include different types of models (e.g., trees and neural networks) as well as different configurations of the same model (e.g., trees with different depths). 


## Creating the Training Set for Stacking {#data-stack}

The first step to building a stacked ensemble relies on the assessment set predictions from a resampling scheme with multiple splits. For each data point in the training set, stacking requires an out-of-sample prediction of some sort. For regression models, this is the predicted outcome. For classification models, the predicted classes or probabilities are available for use, although the latter contains more information than the hard class predictions. For a set of models, a data set is assembled where rows are the training set samples and columns are the out-of-sample predictions from the set of multiple models.

Back in Chapter \@ref(workflow-sets), we used five repeats of 10-fold cross-validation to resample the data. This resampling scheme generates five assessment set predictions for each training set sample. Multiple out-of-sample predictions can occur in several other resampling techniques (e.g. bootstrapping). For the purpose of stacking, any replicate predictions for a data point in the training set are averaged so that there is a single prediction per training set sample per candidate member.

:::rmdnote
Simple validation sets can also be used with stacking since tidymodels considers this to be a single resample. 
:::

For the concrete example, the training set used for model stacking has columns for all of the candidate tuning parameter results. Table \@ref(tab:ensemble-candidate-preds) presents the first six rows and selected columns.


Table: (\#tab:ensemble-candidate-preds)Predictions from candidate tuning parameter configurations.

| Sample # | Bagged Tree | MARS 1 | MARS 2 | Cubist 1 | ... | Cubist 25 | ... |
|:--------:|:-----------:|:------:|:------:|:--------:|:---:|:---------:|:---:|
|    1     |    25.18    | 17.92  | 17.21  |  17.79   |     |   17.82   |     |
|    2     |    5.18     | -1.77  | -0.74  |   2.83   |     |   3.87    |     |
|    3     |    9.71     |  7.26  |  5.91  |   6.31   |     |   8.60    |     |
|    4     |    25.21    | 20.93  | 21.52  |  23.72   |     |   21.61   |     |
|    5     |    6.33     |  1.53  |  0.14  |   3.60   |     |   4.57    |     |
|    6     |    7.88     |  4.88  |  1.74  |   7.69   |     |   7.55    |     |

There is a single column for the bagged tree model since it has no tuning parameters. Also, recall that MARS was tuned over a single parameter (the product degree) with two possible configurations, so this model is represented by two columns. Most of the other models have 25 corresponding columns, as shown for Cubist in this example. 

:::rmdwarning
For classification models, the candidate prediction columns would be predicted class probabilities. Since these columns add to one for each model, the probabilities for one of the classes can be left out. 
:::

To summarize where we are so far, the first step to stacking is to assemble the assessment set predictions for the training set from each candidate model. We can use these assessment set predictions to move forward and build a stacked ensemble.

To start ensembling with the <span class="pkg">stacks</span> package, create an empty data stack using the `stacks()` function and then add candidate models. Recall that we used workflow sets to fit a wide variety of models to these data. We'll use the racing results:


```r
race_results
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result   
#>   <chr>       <list>           <list>    <list>   
#> 1 MARS        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 2 CART        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[3]> <rsmp[+]>
#> 4 RF          <tibble [1 × 4]> <opts[3]> <race[+]>
#> 5 boosting    <tibble [1 × 4]> <opts[3]> <race[+]>
#> 6 Cubist      <tibble [1 × 4]> <opts[3]> <race[+]>
#> # … with 6 more rows
```

In this case, our syntax is:


```r
library(tidymodels)
library(stacks)
tidymodels_prefer()

concrete_stack <- 
  stacks() %>% 
  add_candidates(race_results)

concrete_stack
#> # A data stack with 12 model definitions and 18 candidate members:
#> #   MARS: 1 model configuration
#> #   CART: 1 model configuration
#> #   CART_bagged: 1 model configuration
#> #   RF: 1 model configuration
#> #   boosting: 1 model configuration
#> #   Cubist: 1 model configuration
#> #   SVM_radial: 1 model configuration
#> #   SVM_poly: 1 model configuration
#> #   KNN: 3 model configurations
#> #   neural_network: 1 model configuration
#> #   full_quad_linear_reg: 5 model configurations
#> #   full_quad_KNN: 1 model configuration
#> # Outcome: compressive_strength (numeric)
```

Recall that racing methods (introduced in Chapter \@ref(grid-search)) are more efficient since they might not evaluate all configurations on all resamples. Stacking requires that all candidate members have the complete set of resamples. `add_candidates()` only includes the model configurations that have complete results. 

:::rmdnote
Why use the racing results instead of the full set of candidate models contained in `grid_results`? Either can be used. We found better performance for these data using the racing results. This might be due to the racing method pre-selecting the best model(s) from the larger grid. 
:::

If we had not used the <span class="pkg">workflowsets</span> package, objects from the <span class="pkg">tune</span> and <span class="pkg">finetune</span> could also be passed to `add_candidates()`. This can include both grid and iterative search objects. 

## Blend the Predictions {#blend-predictions}

The training set predictions and the corresponding observed outcome data are used to create a *meta-learning model* where the assessment set predictions are the predictors of the observed outcome data. Meta-learning can be accomplished using any model. The most commonly used model is a regularized generalized linear model, which encompasses linear, logistic, and multinomial models. Specifically, regularization via the lasso penalty [@lasso], which uses shrinkage to pull points toward a central value, has several advantages: 

- Using the lasso penalty can remove candidates (and sometimes whole model types) from the ensemble. 
- The correlation between ensemble candidates tends to be very high and regularization helps alleviate this issue. 

@breiman1996stacked also suggested that, when a linear model is used to blend the predictions, it might be helpful to constrain the blending coefficients to be non-negative. We have generally found this to be good advice and is the default for the <span class="pkg">stacks</span> package (but can be changed via an optional argument). 

Since our outcome is numeric, linear regression is used for the meta-model. Fitting the meta-model is as straightforward as using: 


```r
set.seed(2001)
ens <- blend_predictions(concrete_stack)
```

This evaluates the meta-learning model over a pre-defined grid of lasso penalty values and uses an internal resampling method to determine the best value. The `autoplot()` method, shown in Figure \@ref(fig:stacking-autoplot), helps us understand if the default penalization method was sufficient: 


```r
autoplot(ens)
```

<div class="figure" style="text-align: center">
<img src="figures/stacking-autoplot-1.png" alt="The results of using the `autoplot()` method on the blended stacks object."  />
<p class="caption">(\#fig:stacking-autoplot)Results of using the `autoplot()` method on the blended stacks object.</p>
</div>

The top panel of Figure \@ref(fig:stacking-autoplot) shows the average number of candidate ensemble members retained by the meta-learning model. We can see that the number of members is fairly constant and, as it increases, the RMSE also increases. 

The default range may not have served us well here. To evaluate the  meta-learning model with larger penalties, let's pass an additional option:


```r
set.seed(2002)
ens <- blend_predictions(concrete_stack, penalty = 10^seq(-2, -0.5, length = 20))
```

Now, in Figure \@ref(fig:stacking-autoplot-redo), we see a range where the ensemble model becomes worse than with our first blend (but not by much). The R<sup>2</sup> values increase with more members and larger penalties.


```r
autoplot(ens)
```

<div class="figure" style="text-align: center">
<img src="figures/stacking-autoplot-redo-1.png" alt="The results of using the `autoplot()` method on the updated blended stacks object."  />
<p class="caption">(\#fig:stacking-autoplot-redo)The results of using the `autoplot()` method on the updated blended stacks object.</p>
</div>


It is common, when blending predictions using a regression model, to constrain the blending parameters to be non-negative. For these data, this constraint has the effect of eliminating many of the potential ensemble members; even at fairly low penalties, the ensemble is limited to a fraction of the original eighteen.

The penalty value associated with the smallest RMSE was 0.062. Printing the object shows the details of the meta-learning model: 


```r
ens
#> ── A stacked ensemble model ─────────────────────────────────────
#> 
#> Out of 18 possible candidate members, the ensemble retained 5.
#> Penalty: 0.0615848211066026.
#> Mixture: 1.
#> 
#> The 5 highest weighted members are:
#> # A tibble: 5 × 3
#>   member                    type         weight
#>   <chr>                     <chr>         <dbl>
#> 1 boosting_1_04             boost_tree   0.772 
#> 2 Cubist_1_25               cubist_rules 0.160 
#> 3 full_quad_linear_reg_1_16 linear_reg   0.0445
#> 4 neural_network_1_16       mlp          0.0303
#> 5 MARS_1_2                  mars         0.0134
#> 
#> Members have not yet been fitted with `fit_members()`.
```


The regularized linear regression meta-learning model contained five blending coefficients across five types of models. The `autoplot()` method can be used again to show the contributions of each model type, to produce Figure \@ref(fig:blending-weights). 


```r
autoplot(ens, "weights") +
  geom_text(aes(x = weight + 0.01, label = model), hjust = 0) + 
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.9))
```

<div class="figure" style="text-align: center">
<img src="figures/blending-weights-1.png" alt="fig.alt = &quot;Blending coefficients for the stacking ensemble. The boosted tree and Cubist models have the largest effects on the ensemble predictions.&quot;"  />
<p class="caption">(\#fig:blending-weights)Blending coefficients for the stacking ensemble.</p>
</div>

The boosted tree and Cubist models have the largest contributions to the ensemble. For this ensemble, the outcome is predicted with the equation:


\begin{align}
 \text{ensemble prediction} &=-0.65 \\
	+&0.77 \times \text{boost tree prediction} \notag \\
	+&0.16 \times \text{cubist rules prediction} \notag \\
	+&0.044 \times \text{linear reg prediction} \notag \\
	+&0.03 \times \text{mlp prediction} \notag \\
	+&0.013 \times \text{mars prediction} \notag
\end{align}

where the "predictors" in the equation are the predicted compressive strength values from those models. 

## Fit the Member Models {#fit-members}

The ensemble contains five candidate members and we now know how their predictions can be blended into a final prediction for the ensemble. However, these individual models fits have not yet been created. To be able to use the stacking model, five additional model fits are required. These use the entire training set with the original predictors. 

The five  models to be fit are:

- boosting: number of trees = 1957, minimal node size = 8, tree depth = 7, learning rate = 0.0756, minimum loss reduction = 1.45e-07, and proportion of observations sampled = 0.679

- Cubist: number of committees = 98 and number of nearest neighbors = 2

- linear regression (quadratic features): amount of regularization = 6.28e-09 and proportion of lasso penalty = 0.636

- MARS: degree of interaction = 1

- neural network: number of hidden units = 11, amount of regularization = 0.704, and number of epochs = 692

The <span class="pkg">stacks</span> package has a function, `fit_members()`, that trains and returns these models: 


```r
ens <- fit_members(ens)
```

This updates the stacking object with the fitted workflow objects for each member. At this point, the stacking model can be used for prediction. 

## Test Set Results

Since the blending process used resampling, we can estimate that the ensemble with five members had an estimated RMSE of 4.14. Recall from Chapter \@ref(workflow-sets) that the best boosted tree had a test set RMSE of 3.33. How will the ensemble model compare on the test set? We can `predict()` to find out: 


```r
reg_metrics <- metric_set(rmse, rsq)
ens_test_pred <- 
  predict(ens, concrete_test) %>% 
  bind_cols(concrete_test)

ens_test_pred %>% 
  reg_metrics(compressive_strength, .pred)
#> # A tibble: 2 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard       3.26 
#> 2 rsq     standard       0.958
```

This is moderately better than our best single model. It is fairly common for stacking to produce incremental benefits when compared to the best single model. 

## Chapter Summary {#ensembles-summary}

This chapter demonstrated how to combine different models into an ensemble for better predictive performance. The process of creating the ensemble can automatically eliminate candidate models to find a small subset that improves performance. The <span class="pkg">stacks</span> package has a fluent interface for combining resampling and tuning results into a meta-model. 
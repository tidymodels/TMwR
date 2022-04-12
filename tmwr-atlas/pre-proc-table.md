

# (APPENDIX) Appendix {-} 

# Recommended Preprocessing {#pre-proc-table}

It has been said previously that the type of preprocessing is dependent on the type of model being fit. For example, models that use distance functions or dot products should have all of their predictors on the same scale so that distance is measured appropriately. 

To learn more about each of these models, and others that might be available, see <https://www.tidymodels.org/find/parsnip/>. 

This Appendix provides recommendations for baseline levels of preprocessing that are needed for various model functions. In Table \@ref(tab:preprocessing), the preprocessing methods are categorized as: 

 * **dummy**: Do qualitative predictors require a numeric encoding (e.g. via dummy variables or other methods). 
 
 * **zv**: Should columns with a single unique value be removed? 
 
 * **impute**: If some predictors are missing, should they be estimated via imputation? 
 
 * **decorrelate**: If there are correlated predictors, should this correlation be mitigated? This might mean filtering out predictors, using principal component analysis, or a model-based technique (e.g. regularization). 
  
 * **normalize**: Should predictors be centered and scaled? 
 
 * **transform**: Is it helpful to transform predictors to be more symmetric? 

The information in Table \@ref(tab:preprocessing) is not exhaustive and somewhat depends on the implementation. For example, as noted below the table, some models may not require a particular preprocessing operation but the implementation may require it. In the table, ✓ indicates that the method is required for the model and × indicates that it is not. The ◌ symbol means that the model _may_ be helped by the technique but it is not required.


Table: (\#tab:preprocessing)Preprocessing methods for different models.

|model                          | dummy | zv | impute | decorrelate | normalize | transform |
|:------------------------------|:-----:|:--:|:------:|:-----------:|:---------:|:---------:|
|<tt>bag_mars()</tt>            |   ✓   | ×  |   ✓    |      ◌      |     ×     |     ◌     |
|<tt>bag_tree()</tt>            |   ×   | ×  |   ×    |     ◌¹      |     ×     |     ×     |
|<tt>bart()</tt>                |   ×   | ×  |   ×    |     ◌¹      |     ×     |     ×     |
|<tt>boost_tree()</tt>          |  ×⁺   | ◌  |   ✓⁺   |     ◌¹      |     ×     |     ×     |
|<tt>C5_rules()</tt>            |   ×   | ×  |   ×    |      ×      |     ×     |     ×     |
|<tt>cubist_rules()</tt>        |   ×   | ×  |   ×    |      ×      |     ×     |     ×     |
|<tt>decision_tree()</tt>       |   ×   | ×  |   ×    |     ◌¹      |     ×     |     ×     |
|<tt>discrim_flexible()</tt>    |   ✓   | ×  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>discrim_linear()</tt>      |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>discrim_regularized()</tt> |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>gen_additive_mod()</tt>    |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>linear_reg()</tt>          |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>logistic_reg()</tt>        |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>mars()</tt>                |   ✓   | ×  |   ✓    |      ◌      |     ×     |     ◌     |
|<tt>mlp()</tt>                 |   ✓   | ✓  |   ✓    |      ✓      |     ✓     |     ✓     |
|<tt>multinom_reg()</tt>        |   ✓   | ✓  |   ✓    |      ✓      |    ×⁺     |     ◌     |
|<tt>naive_Bayes()</tt>         |   ×   | ✓  |   ✓    |     ◌¹      |     ×     |     ×     |
|<tt>nearest_neighbor()</tt>    |   ✓   | ✓  |   ✓    |      ◌      |     ✓     |     ✓     |
|<tt>pls()</tt>                 |   ✓   | ✓  |   ✓    |      ×      |     ✓     |     ✓     |
|<tt>poisson_reg()</tt>         |   ✓   | ✓  |   ✓    |      ✓      |     ×     |     ◌     |
|<tt>rand_forest()</tt>         |   ×   | ◌  |   ✓⁺   |     ◌¹      |     ×     |     ×     |
|<tt>rule_fit()</tt>            |   ✓   | ×  |   ✓    |     ◌¹      |     ✓     |     ×     |
|<tt>svm_*()</tt>               |   ✓   | ✓  |   ✓    |      ✓      |     ✓     |     ✓     |

Footnotes: 

1. Decorrelating predictors may not help improve performance. However, fewer correlated predictors can improve the estimation of variance importance scores (see [Fig. 11.4](https://bookdown.org/max/FES/recursive-feature-elimination.html#fig:greedy-rf-imp) of @fes). Essentially, the selection of highly correlated predictors is almost random. 
1. The notation of ⁺ means that the answer depends on the implementation. Specifically: 
  * _Theoretically_, any tree-based model does not require imputation. However, many tree ensemble implementations require imputation. 
  * While tree-based boosting methods generally do not require the creation of dummy variables, models using the `xgboost` engine do. 

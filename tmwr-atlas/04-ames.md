

# (PART\*) Modeling Basics {-} 

# The Ames Housing Data {#ames}

In this chapter, we'll introduce the Ames housing data set [@ames], which we will use in modeling examples throughout this book. Exploratory data analysis, like what we walk through in this chapter, is an important first step in building a reliable model. The data set contains information on 2,930 properties in Ames, Iowa, including columns related to: 

 * house characteristics (bedrooms, garage, fireplace, pool, porch, etc.),
 * location (neighborhood),
 * lot information (zoning, shape, size, etc.),
 * ratings of condition and quality, and
 * sale price.

:::rmdnote
Our modeling goal is to predict the sale price of a house based on other information we have, like its characteristics and location. 
:::

The raw housing data are provided in @ames, but in our analyses in this book, we use a transformed version available in the <span class="pkg">modeldata</span> package. This version has several changes and improvements to the data.^[For a complete account of the differences, see <https://github.com/topepo/AmesHousing/blob/master/R/make_ames.R>.] For example, the longitude and latitude values have been determined for each property. Also, some columns were modified to be more analysis ready. For example: 

 * In the raw data, if a house did not have a particular feature, it was implicitly encoded as missing. For example, there were 2,732 properties that did not have an alleyway. Instead of leaving these as missing, they were relabeled in the transformed version to indicate that no alley was available.

 * The categorical predictors were converted to R's factor data type. While both the tidyverse and base R have moved away from importing data as factors by default, this data type is a better approach for storing qualitative data for modeling than simple strings.  
 * We removed a set of quality descriptors for each house since they are more like outcomes than predictors.

To load the data: 


```r
library(modeldata) # This is also loaded by the tidymodels package
data(ames)

# or, in one line:
data(ames, package = "modeldata")

dim(ames)
#> [1] 2930   74
```

Figure \@ref(fig:ames-map) shows the locations of the properties in Ames. The locations will be revisited in the next section. 

<div class="figure" style="text-align: center">
<img src="premade/ames_plain.png" alt="A scatter plot of house locations in Ames superimposed over a street map. There is a significant area in the center of the map where no homes were sold." width="100%" />
<p class="caption">(\#fig:ames-map)Property locations in Ames, IA.</p>
</div>

The void of data points in the center of Ames corresponds to Iowa State University. 

## Exploring Features of Homes in Ames

Let's start our exploratory data analysis by focusing on the outcome we want to predict: the last sale price of the house (in USD). We can create a histogram to see the distribution of sale prices in Figure \@ref(fig:ames-sale-price-hist).


```r
library(tidymodels)
tidymodels_prefer()

ggplot(ames, aes(x = Sale_Price)) + 
  geom_histogram(bins = 50, col= "white")
```

<div class="figure" style="text-align: center">
<img src="figures/ames-sale-price-hist-1.png" alt="A histogram of the sale prices of houses in Ames, Iowa. The distribution has a long right tail." width="100%" />
<p class="caption">(\#fig:ames-sale-price-hist)Sale prices of houses in Ames, Iowa.</p>
</div>

This plot shows us that the data are right-skewed; there are more inexpensive houses than expensive ones. The median sale price was \$160,000 and the most expensive house was \$755,000. When modeling this outcome, a strong argument can be made that the price should be log-transformed. The advantages of this type of transformation are that no houses would be predicted with negative sale prices and that errors in predicting expensive houses will not have an undue influence on the model. Also, from a statistical perspective, a logarithmic transform may also stabilize the variance in a way that makes inference more legitimate.  We can use similar steps to now visualize the transformed data, shown in Figure \@ref(fig:ames-log-sale-price-hist).


```r
ggplot(ames, aes(x = Sale_Price)) + 
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()
```

<div class="figure" style="text-align: center">
<img src="figures/ames-log-sale-price-hist-1.png" alt="A histogram of the sale prices of houses in Ames, Iowa after a log (base 10) transformation. The distribution, while not perfectly symmetric, exhibits far less skewness." width="100%" />
<p class="caption">(\#fig:ames-log-sale-price-hist)Sale prices of houses in Ames, Iowa after a log (base 10) transformation.</p>
</div>

While not perfect, this will probably result in better models than using the untransformed data, for the reasons we just outlined previously.

:::rmdwarning
The disadvantages to transforming the outcome are mostly related to interpretation of model results.  
:::

The units of the model coefficients might be more difficult to interpret, as will measures of performance. For example, the root mean squared error (RMSE) is a common performance metric that is used in regression models. It uses the difference between the observed and predicted values in its calculations. If the sale price is on the log scale, these differences (i.e. the residuals) are also on the log scale. It can be difficult to understand the quality of a model whose RMSE is 0.15 on such a log scale. 

Despite these drawbacks, the models used in this book utilize the log transformation for this outcome. _From this point on_, the outcome column is pre-logged in the `ames` data frame: 


```r
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```

Another important aspect of these data for our modeling are their geographic locations. This spatial information is contained in the data in two ways: a qualitative `Neighborhood` label as well as quantitative longitude and latitude data. To visualize the spatial information, Figure \@ref(fig:ames-chull) duplicates the data from Figure \@ref(fig:ames-map) with convex hulls around the data from each neighborhood. 

<div class="figure" style="text-align: center">
<img src="premade/ames_chull.png" alt="A scatter plot of house locations in Ames superimposed over a street map with colored regions that show the locations of neighborhoods. Show neighborhoods overlap and a few are nested within other neighborhoods." width="100%" />
<p class="caption">(\#fig:ames-chull)Neighborhoods in Ames represented using a convex hull.</p>
</div>

We can see a few noticeable patterns. First, there is a void of data points in the center of Ames. This corresponds to the campus of Iowa State University where there are no residential houses. Second, while there are a number of neighborhoods that are adjacent to each other, others are geographically isolated. For example, as Figure \@ref(fig:ames-timberland) shows, Timberland is located apart from almost all other neighborhoods.

<div class="figure" style="text-align: center">
<img src="premade/timberland.png" alt="A scatter plot of locations of homes in Timberland, located in the southern part of Ames." width="80%" />
<p class="caption">(\#fig:ames-timberland)Locations of homes in Timberland.</p>
</div>

Figure \@ref(fig:ames-mitchell) visualizes how the Meadow Village neighborhood in Southwest Ames is like an island of properties ensconced inside the sea of properties that make up the Mitchell neighborhood. 

<div class="figure" style="text-align: center">
<img src="premade/mitchell.png" alt="A scatter plot of locations of homes in Meadow Village and Mitchell. The small number of Meadow Village properties are enclosed inside the the ones labeled as being in Mitchell." width="60%" />
<p class="caption">(\#fig:ames-mitchell)Locations of homes in Meadow Village and Mitchell.</p>
</div>
 
A detailed inspection of the map also shows that the neighborhood labels are not completely reliable. For example, Figure \@ref(fig:ames-northridge) shows there are some properties labeled as being in Northridge that are surrounded by homes in the adjacent Somerset neighborhood. 

<div class="figure" style="text-align: center">
<img src="premade/northridge.png" alt="A scatter plot of locations of homes in Somerset and Northridge. There are a few homes in Somerset mixed in the periphery of Northridge (and vice versa)." width="90%" />
<p class="caption">(\#fig:ames-northridge)Locations of homes in Somerset and Northridge.</p>
</div>

Also, there are ten isolated homes labeled as being in Crawford that you can see in Figure \@ref(fig:ames-crawford) but are not close to the majority of the other homes in that neighborhood:

<div class="figure" style="text-align: center">
<img src="premade/crawford.png" alt="A scatter plot of locations of homes in Crawford. There is a large cluster of homes to the west of a small, separate cluster of properties also labeled as Crawford." width="80%" />
<p class="caption">(\#fig:ames-crawford)Locations of homes in Crawford.</p>
</div>

Also notable is the "Iowa Department of Transportation (DOT) and Rail Road" neighborhood adjacent to the main road on the east side of Ames, shown in Figure \@ref(fig:ames-dot-rr). There are several clusters of homes within this neighborhood as well as some longitudinal outliers; the two homes furthest east are isolated from the other locations. 

<div class="figure" style="text-align: center">
<img src="premade/dot_rr.png" alt="A scatter plot of locations of homes labeled as 'Iowa Department of Transportation (DOT) and Rail Road'. The longitude distribution is right-skewed with a few outlying properties." width="100%" />
<p class="caption">(\#fig:ames-dot-rr)Homes labeled as 'Iowa Department of Transportation (DOT) and Rail Road'.</p>
</div>

As previously described in Chapter \@ref(software-modeling), it is critical to conduct exploratory data analysis prior to beginning any modeling. These housing data have characteristics that present interesting challenges about how the data should be processed and modeled. We describe many of these in later chapters. Some basic questions that could be examined during this exploratory stage include: 

 * Are there any odd or noticeable things about the distributions of the individual predictors? Is there much skewness or any pathological distributions? 

 * Are there high correlations between predictors? For example, there are multiple predictors related to the size of the house. Are some redundant?

 * Are there associations between predictors and the outcomes? 

Many of these questions will be revisited as these data are used in upcoming examples. 

## Chapter Summary {#ames-summary}
 
This chapter introduced the Ames housing dataset and investigated some of its characteristics. This data set will be used in later chapters to demonstrate tidymodels syntax. Exploratory data analysis like this is an essential component of any modeling project; EDA uncovers information that contributes to better modeling practice.

The important code for preparing the Ames data set that we will carry forward into subsequent chapters is:
 
 

```r
library(tidymodels)
data(ames)
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```

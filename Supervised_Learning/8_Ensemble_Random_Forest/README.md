# Random Forest

## Task
Predict concrete comprensive strength in MPa (Mega Pascals) given a number of features such as cement, ash, water etc. using random forest regressor. 

## Dataset
The [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) is a collection of measurements of the compressive strength of concrete. It is available on the UCI Machine Learning Repository and contains data from 1030 samples of concrete, with eight input features such as cement, water, and coarse aggregate, and a single output variable, the compressive strength.

Explanation of each variable in the Concrete Compressive Strength Dataset:

1. Cement (kg/m3): The amount of cement used in the concrete mixture, measured in kilograms per cubic meter.

2. Blast Furnace Slag (kg/m3): The amount of blast furnace slag used in the concrete mixture, measured in kilograms per cubic meter.

3. Fly Ash (kg/m3): The amount of fly ash used in the concrete mixture, measured in kilograms per cubic meter.

4. Water (kg/m3): The amount of water used in the concrete mixture, measured in kilograms per cubic meter.

5. Superplasticizer (kg/m3): The amount of superplasticizer used in the concrete mixture, measured in kilograms per cubic meter.

6. Coarse Aggregate (kg/m3): The amount of coarse aggregate (such as gravel or crushed stone) used in the concrete mixture, measured in kilograms per cubic meter.

7. Fine Aggregate (kg/m3): The amount of fine aggregate (such as sand) used in the concrete mixture, measured in kilograms per cubic meter.

8. Age (days): The age of the concrete sample at the time of testing, measured in days.

9. Concrete Compressive Strength (MPa): The compressive strength of the concrete sample, measured in megapascals (MPa). This is the target variable that we are trying to predict.

## The Random Forest Algorithm
Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and stability of predictions. It can be used for both regression and classification tasks.

The key idea behind Random Forest is to build a large number of decision trees and then combine their predictions through voting. Each decision tree is trained on a random subset of the training data and a random subset of the features. This introduces randomness into the model and helps to reduce overfitting.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/random_forest.svg" alt="Random Forest" width="500 height="300">

Image Source: [DeepAI](https://deepai.org/machine-learning-glossary-and-terms/random-forest)

### Working of Random Forest
The Random Forest algorithm works in the following steps:

1. **Random Sampling:** Randomly select "m" samples from the dataset with replacement. This forms the training data for one decision tree.
2. **Random Feature Selection:** Randomly select "p" features from the total "n" features. This is done to reduce the correlation among the trees.
3. **Build Decision Tree:** Build a decision tree using the selected features and samples.
4. **Repeat:** Repeat steps 1 to 3 "k" times to create "k" decision trees.
5. **Prediction:** For a new data point, make predictions using all the "k" decision trees and take a majority vote to get the final prediction.


### Mathematical Formation

The Random Forest algorithm can be mathematically represented as follows:

Let there be "n" observations and "p" features in the dataset. We want to predict the value of a response variable "Y" based on the values of the predictor variables X1, X2, ..., Xp.

**Random Sampling:** Select a random subset of size "m" from the dataset with replacement.

The probability of selecting an observation is given by:

$$p = \frac{1}{n}$$

The probability of not selecting an observation is given by:

$$q = 1 - p$$

The probability of selecting "m" observations with replacement is given by:

$$P(m) = (p^m) * (q^{n-m})$$

**Random Feature Selection:** Select a random subset of size "p" from the total "n" features.

The probability of selecting a feature is given by:

$$p = \frac{1}{p}$$

The probability of not selecting a feature is given by:

$$q = 1 - p$$

The probability of selecting "p" features is given by:

$$P(p) = (p^p) * (q^{n-p})$$

**Build Decision Tree:** Build a decision tree using the selected features and samples.

The decision tree is built using a recursive binary splitting procedure. At each node, the algorithm selects the best feature and split point to minimize the impurity of the two resulting child nodes. The impurity can be measured using various metrics such as Gini index, entropy, or classification error.

**Repeat:** Repeat steps 1 to 3 "k" times to create "k" decision trees.

**Prediction:** For a new data point, make predictions using all the "k" decision trees and take a majority vote to get the final prediction.

The predicted value of "Y" for a new data point can be calculated as follows:

$$Y_{predicted} = mode(Y_1, Y_2, ..., Y_k)$$

where Y_1, Y_2, ..., Y_k are the predicted values of "Y" for the new data point using all the "k" decision trees.


## References/Resources 
- [Concrete Compressive Strength Data Set, UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [Random Forests - DeepAI](https://deepai.org/machine-learning-glossary-and-terms/random-forest)
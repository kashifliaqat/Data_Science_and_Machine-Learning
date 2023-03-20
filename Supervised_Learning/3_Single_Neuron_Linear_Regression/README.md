# Single Neuron Linear Regression

## Task
Predict concrete comprensive strength in MPa (Mega Pascals) given a number of features such as cement, ash, water etc. using single neuron regression from scratch. 

## Dataset
The [Concrete Compressive Strength Data Set](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) is a collection of measurements of the compressive strength of concrete. It is available on the UCI Machine Learning Repository and contains data from 1030 samples of concrete, with eight input features such as cement, water, and coarse aggregate, and a single output variable, the compressive strength.

## Linear Regression
Linear regression is a type of regression analysis that is used to predict a continuous target variable y based on one or more input features X. The model assumes a linear relationship between the input features and the target variable, and tries to fit a linear function to the data. In single neuron linear regression, a single neuron is used to model the relationship between the input variables and the output variable.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/linear_reg.PNG" alt="Linear Regression" width="600" height="250">

Image Source: [Understanding Artificial Neural Network With Linear Regression, AIM](https://analyticsindiamag.com/ann-with-linear-regression/)

### Mathematical Formation
In its simplest form, the linear function can be represented as:

$y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$

where $y$ is the target variable, $x_1$ through $x_n$ are the input features, $w_0$ is the bias term, and $w_1$ through $w_n$ are the weight parameters.

The goal of linear regression is to find the optimal values of the weights that minimize the sum of squared errors between the predicted and actual target values. This is typically done using a method called least squares regression, which involves minimizing the residual sum of squares (RSS) between the predicted and actual target values. 

The RSS is given by:

$$RSS = \sum_{i=1}^{n} (y_{pred}[i] - y[i])^{2}$$

Where $n$ is the number of samples, $y_{pred}[i]$ is the predicted target value for the i-th sample, and $y[i]$ is the actual target value for the $ith$ sample.
#### Optimizing the Model
The optimal values of the weights can be found using a variety of methods, such as gradient descent or normal equations. In this implementation, the gradient descent algorithm is used to find the optimal values of the weights.

In the case of linear regression, the cost function is the RSS between the predicted and actual target values, and is given by:
$$J(w) = \frac{1}{2n} \sum_{i=1}^{n} (y_{pred}[i] - y[i])^{2}$$

Where $n$ is the number of samples, $y_{pred}[i]$ is the predicted target value for the $ith$ sample, $y[i]$ is the actual target value for the $ith$ sample, and $w$ is a vector of weights and the bias term.


- The gradient of the cost function with respect to the weights and bias is given by:
$$\frac{\partial J}{\partial w} = \frac{1}{n} X^{T} (y_{pred} - y)$$
 where $X$ is the feature matrix, $y_{pred}$ is the vector of predicted target values, $y$ is the vector of actual target values, and $T$ denotes the transpose of a matrix.

- The weights and bias are updated using the following rule:

$$w = w - \alpha \frac{\partial J}{\partial w}$$

where $\alpha$ is the learning rate, which determines the step size of the weight and bias updates.

## References/Resources 
- [Concrete Compressive Strength Data Set, UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [Understanding Artificial Neural Network With Linear Regression, AIM](https://analyticsindiamag.com/ann-with-linear-regression/)

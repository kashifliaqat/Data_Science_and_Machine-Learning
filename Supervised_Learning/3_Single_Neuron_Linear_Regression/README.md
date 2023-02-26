# Single Neuron Linear Regression

## Linear Regression
Linear regression is a type of regression analysis that is used to predict a continuous target variable y based on one or more input features X. The model assumes a linear relationship between the input features and the target variable, and tries to fit a linear function to the data. In its simplest form, the linear function can be represented as:

y = w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub>

where w<sub>0</sub>, w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub> are the weights or coefficients of the input features, x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>, and y is the target variable.

- The goal of linear regression is to find the optimal values of the weights that minimize the sum of squared errors between the predicted and actual target values. This is typically done using a method called least squares regression, which involves minimizing the residual sum of squares (RSS) between the predicted and actual target values. 

- The RSS is given by:

    RSS = Σ<sub>i=1</sub><sup>n</sup> (y<sub>pred</sub>[i] - y[i])<sup>2</sup>

    where n is the number of samples, y<sub>pred</sub>[i] is the predicted target value for the i-th sample, and y[i] is the actual target value for the i-th sample.

- The optimal values of the weights can be found using a variety of methods, such as gradient descent or normal equations. In this implementation, I used the gradient descent algorithm to find the optimal values of the weights.


- In the case of linear regression, the cost function is the RSS between the predicted and actual target values, and is given by:

    J(w) = 1/(2*n) * Σ<sub>i=1</sub><sup>n</sup> (y<sub>pred</sub>[i] - y[i])<sup>2</sup>

- where n is the number of samples, y<sub>pred</sub>[i] is the predicted target value for the i-th sample, y[i] is the actual target value for the i-th sample, and w is a vector of weights and the bias term.

- The gradient of the cost function with respect to the weights and bias is given by:

    dJ/dw = 1/n * X<sup>T</sup> * (y<sub>pred</sub> - y)

    where X is the feature matrix, y<sub>pred</sub> is the vector of predicted target values, y is the vector of actual target values, and T denotes the transpose of a matrix.

    The weights and bias are updated using the following rule:

    w = w - lr * dJ/dw

    where lr is the learning rate, which determines the step size of the weight and bias updates.
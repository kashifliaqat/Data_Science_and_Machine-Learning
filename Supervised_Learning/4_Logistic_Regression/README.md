# Logistic Regression

## Task
Implement a single neuron logistic regression from scratch on the breast cancer dataset from Scikit-learn datasets to perform classification between benign or malignant tumors. 

## Dataset
The breast cancer dataset is part of the datasets provided by Scikit-learn. The dataset reports weather a given tumor is benign or malignant based on its characteristics (represented by the various features in the dataset). The goal is to classify whether a given case is malignant (cancerous) or benign (not cancerous) based on these features.

A detailed description and visualization of the cancer dataset is provided in the [Perceptron Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/1_Perceptron/Perceptron.ipynb). 

## The Logistic Regression Algorithm
Logistic regression is a classification algorithm used to predict the probability of a binary outcome (e.g. 0 or 1, yes or no, true or false) given a set of input features. It is a form of supervised learning and is commonly used in machine learning applications such as image recognition, natural language processing, and medical diagnosis. 

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/log_reg.PNG" alt="Logistic Regression" width="700" height="300">

Image Source: [Simplifying Logistic Regression, DZone](https://dzone.com/articles/machinex-simplifying-logistic-regression)

### Working/Implementation of Logistic Regression
To implement logistic regression using neurons, we can use a single output neuron with a sigmoid activation function. The sigmoid function maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

The mathematical equation for a neuron can be expressed as:

$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

where $x_1, x_2, ..., x_n$ are the input features, $w_1, w_2, ..., w_n$ are the corresponding weights, $b$ is the bias term, and $z$ is the weighted sum of the inputs.

The output of the neuron is obtained by applying a non-linear activation function $g$ to the weighted sum $z$:
$$y = g(z)$$

where $g$ is the non-linear activation function and $z$ is the weighted sum of the inputs.

#### Logistic Regression with Neuron
In logistic regression, we want to predict the probability of a binary outcome given a set of input features. Let's say we have $m$ training examples with $n$ input features. We can represent each training example as a vector of $n$ input features, denoted by $\mathbf{x}^{(i)}$. The corresponding binary outcome for each example is denoted by $y^{(i)}$, where $y^{(i)} \in {0,1}$.

We can train the neuron using a cost function known as the cross-entropy loss:

$$ J(\mathbf{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\mathbf{w},b}(\mathbf{x}^{(i)})) + (1-y^{(i)}) \log(1 - h_{\mathbf{w},b}(\mathbf{x}^{(i)})) \right] $$

where $\mathbf{w}$ and $b$ are the weights and bias term of the neuron, $m$ is the number of training examples, $y^{(i)}$ is the true binary outcome of the $ith$ training example, $\mathbf{x}^{(i)}$ is the input feature vector of the $ith$ training example, and $h_{\mathbf{w},b}(\mathbf{x})$ is the output of the neuron given the input vector $\mathbf{x}$ and the parameters $\mathbf{w}$ and $b$.

The goal of training the neuron is to find the values of $\mathbf{w}$ and $b$ that minimize the cost function $J(\mathbf{w},b)$. This can be done using gradient descent, where we update the parameters in the direction of the negative gradient of the cost function:
$$w:=w-\alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$
where $\alpha$ is the learning rate.

Once the parameters have been learned, we can use the neuron to predict the probability of the binary outcome for a new input vector $\mathbf{x}$:

$$\hat{y} = h_{w,b}(x)$$

where $\hat{y}$ is the predicted probability and $h_{\mathbf{w},b}(\mathbf{x})$ is the output of the neuron.

### Summary 
In summary, logistic regression can be implemented using a single output neuron with a sigmoid activation function. The neuron takes a set of input features, applies a weighted sum to them, adds a bias term, and then passes the result through the sigmoid function to obtain the predicted probability of the binary outcome. The neuron can be trained using the cross-entropy loss and gradient descent to find the values of the weights and bias that minimize the cost function. Once the parameters have been learned, the neuron can be used to predict the probability of the binary outcome for a new input vector.


## Resources/References
- [Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Simplifying Logistic Regression, DZone](https://dzone.com/articles/machinex-simplifying-logistic-regression)

# Gradient Descent 

## Task
Implement batch gradient descent from scratch on the breast cancer dataset from Scikit-learn datasets to perform classification between benign or malignant tumors. 

## Dataset
The breast cancer dataset is part of the datasets provided by Scikit-learn. The dataset reports weather a given tumor is benign or malignant based on its characteristics (represented by the various features in the dataset). The goal is to classify whether a given case is malignant (cancerous) or benign (not cancerous) based on these features.

A detailed description and visualization of the cancer dataset is provided in the [Perceptron Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/1_Perceptron/Perceptron.ipynb). 

## The Gradient Descent Algorithm
Gradient descent is an optimization algorithm commonly used in machine learning to find the optimal parameters of a model that minimizes a given cost function. The basic idea of gradient descent is to iteratively update the parameters in the direction of the steepest descent of the cost function, until a minimum is reached.

### Batch Gradient Descent
Batch gradient descent is an optimization algorithm that can be used to minimize a cost function by iteratively adjusting the model parameters in the direction of the negative gradient of the cost function. It is called "batch" gradient descent because it computes the gradient over the entire training set at each iteration.

### Working of Gradient Descent
Gradient Descent is an optimization algorithm used to minimize the cost function of a machine learning model. It works by iteratively adjusting the weights of the model in the opposite direction of the gradient of the Mean Squared Error (MSE) loss function with respect to the weights, in order to find the local minimum of the cost function.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/grad_desc.png" alt="Gradient Descent">

Image Source: [Gradient Descent in Machine Learning, Java T Point](https://www.javatpoint.com/gradient-descent-in-machine-learning)

The algorithm can be expressed mathematically as follows:

1. Initialize the weights of the model with random values.

2. Compute the gradient of the MSE loss function with respect to the weights:

$$\frac{\partial}{\partial w_{j}} J(w) = \frac{2}{m} \sum_{i=1}^{m} (h_{w}(x^{(i)}) - y^{(i)}) x_{j}^{(i)}$$

where:
- $m$ is the number of training examples
- $h_{w}(x^{(i)})$ is the predicted output of the model for the $i^{th}$ training example
- $y^{(i)}$ is the actual output for the $i^{th}$ training example
- $x_{j}^{(i)}$ is the value of the $j^{th}$ feature of the $i^{th}$ training example

3. Update the weights of the model using gradient descent update rule:

$$w_{j} = w_{j} - \alpha \frac{\partial}{\partial w_{j}} J(w)$$


$$w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

where $w$ is the weight vector, $\alpha$ is the learning rate, $m$ is the number of training examples, $x^{(i)}$ is the feature vector of the $ith$ th training example, $y^{(i)}$ is the corresponding label (either 0 or 1), and $h_w(x^{(i)})$ is the predicted label for the $ith$ training example. The weight vector $w$ is updated for each iteration $j$ using the gradient of the cost function with respect to $w_j$.

4. Repeat steps 2-3 until the cost function converges to a minimum or a maximum number of iterations is reached.

#### Summarization
Gradient Descent is an important optimization algorithm used in machine learning to find the local minimum of the cost function. By iteratively adjusting the parameters of the model in the opposite direction of the gradient of the cost function, the algorithm can converge to the optimal values of the parameters. The learning rate is a hyperparameter that needs to be carefully chosen, as a too high learning rate can cause the algorithm to diverge and a too low learning rate can cause the algorithm to converge too slowly.

## Resources/References
[Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) 

[Gradient Descent Algorithm — a deep dive, Towards Datascience](https://towardsdatascience.com/gradient-descent-algorithm-a-deep-dive-cf04e8115f21)

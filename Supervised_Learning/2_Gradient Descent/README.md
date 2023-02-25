# Gradient Descent 
Gradient descent is an optimization algorithm commonly used in machine learning to find the optimal parameters of a model that minimizes a given cost function. The basic idea of gradient descent is to iteratively update the parameters in the direction of the steepest descent of the cost function, until a minimum is reached.

# Batch gradient descent
Batch gradient descent is an optimization algorithm that can be used to minimize a cost function by iteratively adjusting the model parameters in the direction of the negative gradient of the cost function. It is called "batch" gradient descent because it computes the gradient over the entire training set at each iteration.

# Objective
implement batch gradient descent from scratch on the breast cancer dataset to perform classification.

# Implementation

## The Breast Cancer Dataset 
The breast cancer dataset is a classification dataset that contains information about breast cancer cases, including various features such as the radius, texture, and perimeter of the cell nuclei. The goal is to classify whether a given case is malignant (cancerous) or benign (not cancerous) based on these features.

## The Cost Function
The cost function is a measure of how well the model is able to classify the breast cancer cases. We used the mean squared error (MSE) as our cost function:

$$J(w) = \frac{1}{2m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)})^2$$ 

where $w$ is the weight vector, $m$ is the number of training examples, $x^{(i)}$ is the feature vector of the $i$th training example, $y^{(i)}$ is the corresponding label (either 0 or 1), and $h_w(x^{(i)})$ is the predicted label for the $i$th training example.

## The gradient descent update rule is:

$$w_j = w_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_w(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

where $w$ is the weight vector, $\alpha$ is the learning rate, $m$ is the number of training examples, $x^{(i)}$ is the feature vector of the $i$th training example, $y^{(i)}$ is the corresponding label (either 0 or 1), and $h_w(x^{(i)})$ is the predicted label for the $i$th training example. The weight vector $w$ is updated for each iteration $j$ using the gradient of the cost function with respect to $w_j$.



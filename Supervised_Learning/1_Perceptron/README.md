# Perceptron

## Task
Perform binary classification on cancer dataset from Scikit-learn by building a perceptron model from scratch. 

## Dataset
The breast cancer dataset is part of the datasets provided by Scikit-learn. The dataset reports weather a given tumor is benign or malignant based on its characteristics (represented by the various features in the dataset). 

A detailed description and visualization of the cancer dataset is provided in the [Perceptron Notebook](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/1_Perceptron/Perceptron.ipynb). 

## The Perceptron
The perceptron is a single-layer neural network that can be used for binary classification tasks, meaning it can classify inputs into one of two categories.

The perceptron is inspired by the structure and function of biological neurons in the human brain. Like biological neurons, artificial neurons in a perceptron receive input signals from other neurons, process these signals, and produce output signals.

The perceptron consists of one or more input units, a bias unit, and a single output unit, as shown in the following diagram:
<img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/perceptron.png" alt="The Perceptron">

Image Source: [Keras Tutorial: Deep Learning in Python](https://www.datacamp.com/tutorial/deep-learning-python) 

Each input unit is connected to the output unit with a weight, and the bias unit is also connected to the output unit with its own weight. The output unit uses a step function to produce a binary output based on the weighted sum of the inputs and bias.

### Working of Perceptron
The output of the perceptron is determined by the following formula:

$$w_i = w_i + \text{learning_rate} \times (target - output) \times x_i$$


where:

- $x_1$, $x_2$, $\ldots$, $x_n$ are the input features
- $w_1$, $w_2$, $\ldots$, $w_n$ are the corresponding weights for each input feature
- $w_0$ is the bias weight
- $\text{step}$ is the step function: The step function returns 1 if the weighted sum of the inputs and bias is greater than or equal to 0, and 0 otherwise. In other words, the perceptron outputs a 1 if the input falls on one side of the decision boundary and a 0 if it falls on the other side.

The perceptron learns by adjusting the weights based on the error between the predicted output and the true output. The weights are updated using the following formula:

$$w_i = w_i + \text{learning_rate} \times (target - output) \times x_i$$


where:

- $w_i$ is the weight for the current input feature $x_i$
- $learning_-rate$ is the hyperparameter that determines the step size for adjusting the weights
- $target$ is the desired output for the given input
- $output$ is the actual output produced by the perceptron for the given input
- $x_i$ is the value of the current input feature being considered for weight update

The perceptron is a simple but powerful algorithm that can be used for binary classification tasks. 

## Resources/References
[Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) 

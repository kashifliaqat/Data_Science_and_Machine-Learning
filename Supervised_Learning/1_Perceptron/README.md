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
<img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/perceptron.png" alt="Machine Learning">
Image Source: [Keras Tutorial: Deep Learning in Python](https://www.datacamp.com/tutorial/deep-learning-python) 

Each input unit is connected to the output unit with a weight, and the bias unit is also connected to the output unit with its own weight. The output unit uses a step function to produce a binary output based on the weighted sum of the inputs and bias.

### Working of Perceptron

w_i = w_i + learning_rate * (target - output) * x_i

y = \text{step}(w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n)




## Resources/References
[Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) 
# Supervised Learning
Supervised Learning involves training a model on labeled data to make predictions on unseen or future data. In supervised learning, the input data is accompanied by the correct output, and the algorithm learns to map the input to the output by minimizing a loss function.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/supervised_learning.PNG" alt="Supervised Learning" width="600" height="300">

Image Source: [Symmetry (MDPI)](https://www.mdpi.com/2073-8994/10/12/734)

## Mathematical Formulation
Suppose we have a dataset with $n$ samples and $m$ features. In supervised learning, each sample $i$ has an input feature vector $\boldsymbol{x}_i$ and an output label $y_i$. We can represent the dataset as a set of pairs ${( \boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_2),..., (\boldsymbol{x}_n, y_n)}$.

The goal of supervised learning is to learn a function $f(\boldsymbol{x})$ that maps input feature vectors $\boldsymbol{x}$ to their corresponding output labels $y$. The function $f(\boldsymbol{x})$ is often represented as a model with parameters $(\boldsymbol{w})$. We can find the optimal values of $(\boldsymbol{w})$ by minimizing a loss function $L(\boldsymbol{w})$ that measures the difference between the predicted output and the actual output.

The most commonly used loss functions in supervised learning are mean squared error (MSE), cross-entropy loss, and hinge loss. The choice of loss function depends on the problem at hand and the type of output variable.

## Types of Supervised Learning
There are two types of supervised learning: regression and classification.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/reg_vs_class.PNG" alt="Regression vs Classification" width="600" height="340">

Image Source: [Regression vs Classification, Explained, SHARP SIGHT](https://www.sharpsightlabs.com/blog/regression-vs-classification/)

### Regression
Regression is a supervised learning problem where the output variable is continuous. The goal of regression is to learn a function that predicts a continuous output variable given one or more input variables.

- Examples of regression problems include predicting housing prices, stock prices, or the amount of rainfall in a given area.

### Classification
Classification is a supervised learning problem where the output variable is discrete. The goal of classification is to learn a function that predicts a discrete output variable given one or more input variables.

- Examples of classification problems include predicting whether an email is spam or not, predicting the species of a flower based on its characteristics, or predicting whether a customer will buy a product or not.

## Supervised Learning Models 
The following is the list of supervised learning algorithms developed and discussed in this repo. 

1. [Perceptron](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/1_Perceptron)
2. [Gradient Descent](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/2_Gradient%20Descent) 
3. [Single Neuron Linear Regression](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/3_Single_Neuron_Linear_Regression)
4. [Logistic Regression](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/4_Logistic_Regression)
5. [Deep Neural Network](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/5_Deep_Neural_Network)
6. [K-Nearest Neighbors](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/6_k_Nearest_Neighbors)
7. [Decision Trees](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/7_Decision_Trees)
8. [Ensemble Learning - Random Forest](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/8_Ensemble_Random_Forest)
9. [Support Vector Machines](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Supervised_Learning/9_Support_Vector_Machines)

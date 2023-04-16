# Support Vector Machines (SVMs)

## Task
Predict concrete comprensive strength in MPa (Mega Pascals) given a number of features such as cement, ash, water etc. using SVM. 

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

## The Support Vector Machine Algorithm

Support Vector Machines (SVMs) are a set of supervised learning methods used for classification, regression, and outliers detection. SVM works by mapping the data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/svm.PNG" alt="Support Vector Machines" width="500 height="300">
<p align="center"><em> SVM: Regression vs Classification.</em></p>

Image Source: [Medium](https://medium.com/it-paragon/support-vector-machine-regression-cf65348b6345)

### SVM for Classification
SVM is a binary classification algorithm, which means it can classify data points into two classes only. However, we can use SVM to solve multiclass classification problems by using a combination of binary classifiers. The goal of SVM is to find a hyperplane that separates the two classes in a way that maximizes the margin between them. The margin is the distance between the hyperplane and the closest data points from each class. The hyperplane is defined by the equation:

$$
w^Tx + b = 0
$$

Where $w$ is the weight vector, $x$ is the input vector, and $b$ is the bias. The sign of the expression $w^Tx + b$ determines the class to which the input vector belongs. If it is positive, the input vector belongs to one class, and if it is negative, the input vector belongs to the other class.

To find the hyperplane that maximizes the margin, we need to solve the following optimization problem:

$$
\min_{w,b} \frac{1}{2}||w||^2 \\
\text{s.t. } y_i(w^Tx_i + b) \geq 1 \text{ for } i=1,\ldots,n
$$

where $y_i$ is the class label of the $i$-th data point. The optimization problem aims to minimize the norm of the weight vector $w$ subject to the constraint that all data points are correctly classified with a margin of at least 1.

The solution to this problem is obtained using Lagrange multipliers. The optimization problem can be converted into its dual form, which leads to the following equation for the hyperplane:

$$
f(x) = \sum_{i=1}^{n}\alpha_i y_i \langle x_i,x \rangle + b
$$

where $\alpha_i$ are the Lagrange multipliers and $\langle x_i,x \rangle$ is the dot product between the $i$-th data point and the input vector. The SVM algorithm only uses a subset of the data points, called support vectors, to compute the hyperplane. The support vectors are the data points that lie on the margin or are misclassified.

### SVM for Regression
SVM can also be used for regression tasks, where the goal is to predict a continuous target variable instead of a class label. In this case, SVM aims to find a hyperplane that best fits the data points with a minimum margin violation. The hyperplane is defined by the same equation as in the classification case:

$$
f(x) = w^Tx + b
$$

To find the best hyperplane, we need to solve the following optimization problem:

$$
\min_{w,b,\epsilon} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n} \epsilon_i \\
\text{s.t. } |y_i - f(x_i)| \leq \epsilon_i \text{ for } i=1,\ldots,n
$$

Where $C$ is a hyperparameter that controls the trade-off between the margin violation and the magnitude of the weight vector, and $\epsilon_i$ are the slack variables that allow for some margin violation. The optimization problem aims to minimize the norm of the weight vector $w$ subject to the constraint that the distance between the predicted value $f(x_i)$ and the true value $y_i$ is less than or equal to $\epsilon_i$.

## References/Resources 
- [Concrete Compressive Strength Data Set, UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)
- [Support Vector Machine: Regression - Medium](https://medium.com/it-paragon/support-vector-machine-regression-cf65348b6345)
# Principal Component Analysis (PCA)

## Task
Use PCA to reduce the dimensionality of the Wine dataset and then use the reduced dataset to train a classifier to predict the class of wine. Show the effect of PCA on the accuracy of the classifier.

## Dataset
The [Wine Quality Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#) is a popular machine learning dataset used for classification tasks. It contains information on 178 different wines, each with 13 different features. The dataset is commonly used to classify wines into one of three different classes based on their chemical characteristics.

Here are the variables in the Wine dataset:

- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

The target variable in this dataset is the wine class, which is one of three possible values: class_0, class_1, or class_2. The dataset can be loaded using the `load_wine()` function from the scikit-learn library in Python.

These classes represent different cultivars of the wine, namely:

- class_0: 'Barolo' wines
- class_1: 'Grignolino' wines
- class_2: 'Barbera' wines 

## The PCA Algorithm

PCA is a popular dimensionality reduction technique that is widely used in machine learning, statistics, and data analysis. It is used to reduce the number of features in a dataset by transforming the original features into a lower-dimensional space, while preserving the most important information.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/pca.PNG" alt="DBSCAN" width="500 height="300">


Image Source: [
Towards Data Science](https://towardsdatascience.com/understanding-principal-component-analysis-ddaf350a363a)

## How does PCA work?

PCA works by finding the principal components of a dataset, which are the directions in the feature space along which the data varies the most. These principal components are the linear combinations of the original features, and they are orthogonal to each other (i.e., they are perpendicular). The first principal component captures the most variance in the data, the second principal component captures the second most variance, and so on.

To find the principal components, PCA performs the following steps:

1. Center the data by subtracting the mean from each feature.
2. Compute the covariance matrix of the centered data.
3. Compute the eigenvectors and eigenvalues of the covariance matrix.
4. Sort the eigenvectors by their corresponding eigenvalues, in descending order.
5. Select the top k eigenvectors with the highest eigenvalues, where k is the desired number of dimensions in the reduced space.
6. Transform the data into the reduced space by multiplying it by the k eigenvectors.

## Mathematical formulation of PCA

Let X be an n x m data matrix, where n is the number of samples and m is the number of features. PCA aims to find a transformation matrix W that maps X onto a new space Y, such that the dimensions of Y are lower-dimensional than the dimensions of X.

The transformation matrix W is computed by finding the eigenvectors of the covariance matrix S, which is given by:

$$S = (1/n) * X^T * X$$

where $X^T$ is the transpose of $X$. The eigenvectors of $S$ are denoted by $w_1$, $w_2$, ..., $w_m$, and the corresponding eigenvalues are denoted by $λ_1$, $λ_2$, ..., $λ_m$.

The first k principal components are given by the first k eigenvectors, which have the highest eigenvalues. To compute the reduced data matrix Y, we multiply X by the transformation matrix W, which is given by:

$$W = [w_1, w_2, ..., w_k]$$

$$Y = X * W$$

where $Y$ is an $n$ x $k$ matrix containing the transformed data, and $k$ is the desired number of dimensions in the reduced space.


## References

- [Wine Quality Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#)

- [Understanding Principal Component Analysis- Towards Data Science](https://towardsdatascience.com/understanding-principal-component-analysis-ddaf350a363a)
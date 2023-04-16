# K-Nearest Neighbors

## Tasks
1. Use K-Nearest Neighbors to perform classification to predict whether a patient has diabetes or not.

## Dataset
The [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) is a collection of medical data from female patients of Pima Indian heritage. The dataset contains information on various medical variables, including glucose level, blood pressure, and body mass index, and their relation to the onset of diabetes in the patients.

The dataset consists of 9 variables:

1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age in years
9. Outcome: Class variable (0 or 1)  indicating whether or not the patient developed diabetes within 5 years of the initial examination. A value of 1 indicates that the patient developed diabetes, while a value of 0 indicates that they did not develop diabetes.

## The K-Nearest Neighbors Algorithm

K-Nearest Neighbor (KNN) is a non-parametric and supervised machine learning algorithm that is used for both classification and regression problems. It works by finding the k-nearest neighbors of a given query data point from the training dataset and then uses the majority vote (in classification) or the average value (in regression) of the k neighbors to predict the label or value of the query data point.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/knn.png" alt="K-Nearest Neighbors" width="500" height="300">

<p align="left"><em> New data point must belong to category A as  3 nearest neighbors are from category A.</em></p>

Source for Images: [javaTpoint](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)

### How KNN Works
1. **Data Preparation:** Start by cleaning and preprocessing the data by removing missing values and scaling the features to ensure they are on the same scale.

2. **Choose the Value of K:** Select the value of k (the number of neighbors) based on the size of the dataset and the problem at hand. A small k value will lead to more flexible decision boundaries, while a large k value will lead to more rigid decision boundaries.

3. **Calculate Distances:** Calculate the distance between the query data point and all the training data points using a distance metric such as Euclidean, Manhattan, or Minkowski distance.

4. **Find K-Nearest Neighbors:** Select the k-nearest neighbors of the query data point based on the calculated distances.

5. **Majority Vote (Classification) or Average (Regression):** For classification problems, the label of the query data point is predicted based on the majority vote of the k-nearest neighbors. For regression problems, the value of the query data point is predicted based on the average value of the k-nearest neighbors.

6. **Evaluate the Model:** Finally, evaluate the performance of the model by using metrics such as accuracy, precision, recall, or mean squared error, depending on the problem at hand.

#### Distance Metrics
1. **Euclidean Distance:** The Euclidean distance between two data points x and y in n-dimensional space is given by: 

<div style="text-align:center">

$$
\sqrt{(x_1-y_1)^2+(x_2-y_2)^2+...+(x_n-y_n)^2}
$$

</div>



2. **Manhattan Distance:** The Manhattan distance between two data points x and y in n-dimensional space is given by:

<div style="text-align:center">

$$
||x_1-y_1||+||x_2-y_2||+...+||x_n-y_n||
$$

</div>

3. **Minkowski Distance:** The Minkowski distance between two data points x and y in n-dimensional space is given by: 

<div style="text-align:center">

$$
\left(\sum_{i=1}^n|{x_i-y_i}|^p\right)^{\frac{1}{p}}
$$

</div>

Where p is a parameter that determines the order of the Minkowski distance. When p=1, the Minkowski distance is equivalent to the Manhattan distance, and when p=2, it is equivalent to the Euclidean distance.

## Resources/References
1. [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
2. [K-Nearest Neighbors Algorithm for ML - Serokell.io](https://serokell.io/blog/knn-algorithm-in-ml)

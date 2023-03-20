# The Multilayer Perceptron (Deep Neural Network)

## Tasks
1. Build a multilayer feedforward network from scratch and implement for the Fashion MNIST classification problem.
    - Notebook: [Task1_Deep_NN_scratch](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task1_Deep_NN_scratch.ipynb)
2. Build a multilayer feedforward network using `tensorflow` and `keras` for the Fashion MNIST classification problem.
    - Notebook: [Task2_Deep_NN_Tensorflow](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task2_Deep_NN_Tensorflow.ipynb)

## Dataset
The [Fashion MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist) is a popular benchmark dataset in machine learning and computer vision research. It consists of 70,000 grayscale images of fashion products, divided into 10 categories. Each image is 28x28 pixels in size.

The 10 categories in the dataset are as follows:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is intended as a drop-in replacement for the original MNIST dataset, which contains handwritten digits. Fashion MNIST is more challenging than MNIST, as the images are more complex and there is more variability within each category.

- More information and data visualization is provided in [Task1_Deep_NN_scratch](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/blob/main/Supervised_Learning/5_Deep_Neural_Network/Task1_Deep_NN_scratch.ipynb).

## The Multilayer Perceptron
A multilayer feedforward network, also known as a multilayer perceptron (MLP), is a type of artificial neural network (ANN) that consists of multiple layers of nodes, each connected to the next. The nodes in each layer are fully connected to the nodes in the next layer, but there are no connections between nodes within the same layer. The following figures provides an examples of MLP architecture.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/NN_gif.gif" alt="1 Hidden MultiLayer Perceptron" width="500" height="300">

<p align="center"><em>MLP with one hidden layer.</em></p>

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/MLP.png" alt="MultiLayer Perceptron" width="500" height="300">

<p align="center"><em>MLP with two hidden layers.</em></p>

Source for Images: [Medium](https://miro.medium.com/v2/resize:fit:1000/1*3fA77_mLNiJTSgZFhYnU0Q.png)

### Mathematical Formation
The mathematical formulation of an MLP can be described as follows. Let $X$ be the input to the network, and let $Y$ be the desired output. The network consists of $L$ layers, where the first layer is the input layer, the last layer is the output layer, and the remaining layers are hidden layers. The output of the $jth$ node in the $ith$ layer is denoted by $h_i^j$. The weights connecting the $jth$ node in layer $i$ to the $kth$ node in layer $i+1$ are denoted by $w_i^j,k$, and the biases of the $kth$ node in layer $i+1$ are denoted by $b_i^k$.

To compute the output of the network, we first apply the following transformation to the input:

$h_1^j = x_j$

For each subsequent layer $i$, we compute the weighted sum of the outputs of the previous layer, and apply a non-linear activation function $g_i$ to obtain the output of each node:

$h_i^j = g_i(\sum_{k=1}^{n_{i-1}}w_{i-1}^k,jh_{i-1}^k + b_{i-1}^j)$

where $n_{i-1}$ is the number of nodes in layer $i-1$, and $j$ and $k$ are indices over the nodes in layer $i$ and layer $i-1$, respectively.

The output of the network is given by the output of the final layer:

$y_k = h_L^k$

The parameters of the network, namely the weights and biases, are learned by minimizing a loss function. A common choice of loss function is the mean squared error (MSE), which is defined as:

$MSE(Y, \hat{Y}) = \frac{1}{N}\sum_{i=1}^N\sum_{k=1}^K (y_i^k - \hat{y_i^k})^2$

where $N$ is the number of samples in the dataset, $K$ is the number of output classes, $y_i^k$ is the true label of the $ith$ sample for the $kth$ class, and $\hat{y_i^k}$ is the predicted label of the $ith$ sample for the $kth$ class.

The parameters of the network are updated using stochastic gradient descent (SGD), which involves computing the gradient of the loss function with respect to the parameters and updating the parameters in the opposite direction of the gradient. The update equation for the weights and biases are given by:

$w_{i}^j,k \leftarrow w_{i}^j,k - \alpha\frac{\partial L}{\partial w_{i}^j,k}$

$b_{i}^k \leftarrow b_{i}^k - \alpha\frac{\partial L}{\partial b_{i}^k}$

where $\alpha$ is the learning rate, and $\frac{\partial L}{\partial w_{i}^j,k}$ and $\frac{\partial L}{\partial b_{i}^k}$ are the gradients of the loss function with respect to the weights and biases, respectively, which can be computed using backpropagation, a recursive algorithm for computing the gradients of the loss function with respect to the parameters.

The backpropagation algorithm starts by computing the gradient of the loss function with respect to the output layer:

$\frac{\partial L}{\partial h_L^k} = \frac{1}{N}(y_i^k - \hat{y_i^k})$

For each subsequent layer $i$ in reverse order, we compute the gradients of the outputs with respect to the weighted sum and the gradients of the weighted sum with respect to the weights and biases:

$\frac{\partial h_i^j}{\partial z_i^j} = g_i'(z_i^j)$

$\frac{\partial z_i^j}{\partial w_{i}^j,k} = h_{i-1}^k$

$\frac{\partial z_i^j}{\partial b_{i}^k} = 1$

where $g_i'$ is the derivative of the activation function $g_i$.

Finally, we compute the gradient of the loss function with respect to the weights and biases using the chain rule:

$\frac{\partial L}{\partial w_{i}^j,k} = \frac{\partial L}{\partial h_i^j}\frac{\partial h_i^j}{\partial z_i^j}\frac{\partial z_i^j}{\partial w_{i}^j,k}$

$\frac{\partial L}{\partial b_{i}^k} = \frac{\partial L}{\partial h_i^j}\frac{\partial h_i^j}{\partial z_i^j}\frac{\partial z_i^j}{\partial b_{i}^k}$

Using these gradients, we can update the weights and biases using the SGD update equation mentioned earlier.

By adjusting the hyperparamters such as number of layers, the number of nodes in each layer, and the choice of activation function, we can design networks that can capture complex patterns in the data and achieve high levels of accuracy on various tasks.

## Resources/References
1. [Fashion MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
2. [Understanding Neural Networks - Medium](https://prince-canuma.medium.com/understanding-neural-networks-22b29755abd9)

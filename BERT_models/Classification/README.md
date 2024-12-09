# Classification with BERT Models
* This repo contains projects and experiments with various BERT models for classification tasks in NLP.





## Review of Loss Functions and Activation Functions
* This is an excellent chart to review for loss functions and activation functions in classification tasks (source: Baruah, 2021)

![image](https://github.com/user-attachments/assets/845ad5c2-0cfc-43f3-9da9-59bb783975e7)


### Activation Functions
* This is a function which obtains the output from a node in neural network such as yes or no(0 to 1) depending on the function).
* The purpose of an activation function is to add **non-linearity** to the neural network.
* As an example:
  * Let’s say we have a neural network without activation functions.
  * In such a case, **every layer** will only be performing a **linear transformation** on the inputs using the weights and biases as `net input =∑(weight * input) + bias`.
  * It will not matter how many hidden layers we attach to the neural network, **all layers will behave in the same way** because the **result of two linear functions is a linear function itself!**

* In a neural network, **all hidden layers typically use the same activation function**.
* It is the **output layer** that will use a **different activation function from the hidden layers** and is **dependent upon the type of prediction required by the model**.
* Activation functions are also **differentiable**.
    * This means the first-order derivative can be calculated for a given input value.
    * This is required given that neural networks are trained using the backpropagation of error algorithm that requires the derivative of prediction error in order to update the weights of the model.

#### Non-linear activation functions
1. **Sigmoid or Logistic Activation Function**
   * S shaped function that takes any real value as input and outputs values in the range of 0 to 1.
   * **Most commonly used where we have to predict the probability as an output.**
   * The function is differentiable which means we can find the slope of the sigmoid curve at any two points.
   * The derivative of sigmoid function is **not symmetric around zero and suffers from a vanishing gradient.**
  
![image](https://github.com/user-attachments/assets/78a5b7d9-be4d-4b22-98e6-31d1e5374ca4)
* source: Sheehan github (see references)

  
2. **Tanh or Hyperbolic Tangent Function**
   * Very similar to the sigmoid/logistic activation function with the **difference in output range of -1 to 1.**
   * The advantage is that the **negative inputs will be mapped strongly negative** and the **zero inputs will be mapped near zero** in the tanh graph.
   * Similar to sigmoid, its derivate also faces the problem of **vanishing gradients**.
   * In addition, the gradient of the tanh function is much steeper(is differentiable) as compared to the sigmoid function.

![image](https://github.com/user-attachments/assets/7bf4ce07-ecee-4f40-bc82-26ddffc1fed6)
* source: Sheehan github (see refs)


3. **RELU Function**
   * ReLU stands for "Rectified Linear Unit."
   * The ReLU function **does not activate all the neurons at the same time.**
   * The neurons will **only be deactivated if the output of the linear transformation is less than 0.**
   * ReLU accelerates the convergence of gradient descent towards the global minimum of the loss function due to its linear, non-saturating property.
   * The negative side of the graph makes the gradient value zero.
        * Due to this problem, during the backpropagation process, the weights and biases for some neurons **are not updated**, also known as the **Dying ReLu problem**.
        * This can create **dead neurons which never get activated.**
    
![image](https://github.com/user-attachments/assets/2b81a517-14e8-440d-8afa-938c5b09eff0)
* source: Sheehan github (see refs)

    
4. **Leaky RELU Function**
   * Leaky ReLU is an improved version of ReLU function to solve the Dying ReLU problem as it has a small positive slope in the negative area.
   * It enables backpropagation, even for negative input values.
   * However, the **predictions may not be consistent for negative input values.**
  
![image](https://github.com/user-attachments/assets/afedac09-944c-4d1f-aea9-30a4f1d89ca4)
* source: Sheehan github (see refs)

  
5. **Parametric RELU Function**
   * Another variant of ReLU that aims to solve the problem of gradient’s becoming zero for the left half of the axis.
   * This function provides the slope of the negative part of the function as an **argument a**.
   * By performing backpropagation, the **most appropriate value of a is learnt.**
  
![image](https://github.com/user-attachments/assets/44f7e8b9-4800-43b3-bcd8-524b2ca96465)
* source: Sheehan github (see refs)

  
#### Review of Activation Function Usage 
1. **Hidden layer Activation Functions**
   * Modern neural network models with common architectures, such as MLP and CNN, will usually use the **ReLU activation function**, or extensions.
   * Recurrent networks (RNNs) still commonly use **Tanh or sigmoid activation functions**, or even both.

2. **Output layer Activation Functions**
   * **Regression** One node, linear activation.
   * **Binary Classification**: One node, sigmoid activation.
   * **Multiclass Classification**: One node per class, softmax activation.
   * **Multilabel Classification**: One node per class, sigmoid activation.




### Loss Functions
* A method of evaluating how well your algorithm/model is modeling your dataset.
* It is used to optimize the model during training.
* Based on the your task, loss functions are classified as follows:

**1. Regression Loss**
  * Mean Square Error or L2 Loss
  * Mean Absolute Error or L1 Loss
  * Huber Loss

**2. Classification Loss**
  * **Binary Classification**
      * Hinge Loss
         * Mostly used with Support Vector Machine (SVM) models in machine learning.
         * Target variable must be modified to be in the set of {-1,1}.
      * Squared hinge loss
         * Used for “maximum margin” binary classification problems.
         * It has the effect of smoothing the surface of the error function and making it numerically easier to work with.
         * Target variable must be modified to be in the set of {-1,1}.
      * Sigmoid Cross Entropy (Logistic) Loss
         * Transform x-values by the sigmoid function before applying the cross entropy loss.
         * **Cross entropy loss is also known as logistic loss function.**
         * It’s the most common loss for binary classification (two classes 0 and 1).
         * When we want to measure the distance from the actual class to the predicted value, which is usually a real number between 0 and 1, we use the sigmoid activation function as the last layer before applying CE. 
      * Weighted Cross Entropy Loss
         * Weighted version of the sigmoid cross entropy loss.
         * You provide a weight on the positive target to control the outliers for positive predictions.

  * **Multi-Class Classification**
      * Softmax Categorical Cross Entropy Loss
      * Sparse Categorical Cross Entropy Loss
      * Kullback-Leibler Divergence Loss


#### Summary of Loss Functions
* Source: Huynh, 2023
![image](https://github.com/user-attachments/assets/98771119-b3c6-430d-a900-4e5fec87c8dd)


### Optimizers
* An optimizer is a function or an algorithm that modifies the attributes of a neural network, such as weights and learning rate.
* Thus, it helps in reducing the overall loss and improve the accuracy and can help reduce training time exponentially.
* These are solvers of minimization problems in which the function to minimize has a gradient.

#### Types of Optimizers

1. **Gradient Descent**
    * Batch gradient descent
    * Stochastic gradient descent (SGD)
    * Mini-batch gradient descent
  
  * **Summary of Gradient Descent Optimizers**
      * Mini batch gradient descent is the best choice among the three in most use cases.
      * Learning rate tuning problem: all are subject to the choice of a good learning rate. Unfortunately, this choice is not intuitive. 
      * Generally speaking these optimizers are NOT good for sparse data: there is no mechanism to put in evidence rarely occurring features. All parameters are updated equally.
      * Overall there is a High possibility of getting stuck into a suboptimal local minima.


2. **Adaptive**
    * Adagrad
    * Adadelta
    * RMSprop
    * Adam
  
* **Summary of Adaptive Optimizers**
      * **Adam is considered the best among the adaptive optimizers in most use cases.**
      * **Overall Adaptive optimizers are excellent with sparse data**: the adaptive learning rate is perfect for this type of datasets.
      * With these optimizers there is no need to focus on the learning rate value.


#### Gradient Descent vs. Adaptive Optimizers
* Adam is the default best choice for most modeling use cases.
* Recent papers state that SGD can bring to better results if combined with a good learning rate annealing schedule which aims to manage its value during the training.
* **Usually the best approach is to start with the Adam optimizer -- this is because it is more likely to return good results without any advanced fine tuning.**
  * **If Adam returns good results, it might be a good idea to then try SGD just to see what happens, it never hurts.**




# References
* (Sheehan, D. Visualising Activation Functions in Neural Networks)[https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks/]

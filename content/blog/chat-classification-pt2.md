---
title: "Group Chat Classificiation Part 2: Model Evaluation and Naive Classifier"
date: 2023-08-23
draft: True
---
In the [previous blog](../chat-classification-pt1), we loaded in my group chat messages and looked at some of the trends in the data. Before we can start chugging away on models, we have to answer the question: How do we know if we built a good model? There are many ways to answer this question. You can go really deep down the rabbit hole of measuring model quality (just ask ChatGPT), but for the sake of this post we will use the basic measure of [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy). Accuracy is defined as the proportion of the time you correctly classify the input. 

In our case, this means the proportion of time that we correctly identify who sent a message without knowing who that is beforehand. For example, if our model correctly predicted 4 messages' senders and incorrectly predicted 6 messages' senders, our accuracy would be 4 / (4+6) = 40%.

<!-- more -->

### Building a Classifier Evaluator
As mentioned above, accuracy can be calculated as the number of correct predictions divided by the total number of predictions. There are multiple ways we could represent this in code (each potential class could be represented as a string, each class could be coded as an integer, etc.), but we will take the common Machine Learning approach of vectorizing both the predictions and ground truth. The ground truth will be encoded using [one-hot encoding](https://en.wikipedia.org/wiki/One-hot), where each potential class is an element in a vector (list of integers) where all values are 0 except for the correct output, which is 1. For example if we have a list of senders Bob, Fred, and John and a particular message was sent by John, the one hot vector may look like: `[0, 0, 1]`.

Vectorizing the inputs will be simple as most models we work with in this series will be probabilistic models that output a probability of each class being the correct class. In the example above, if we had a good model it might output `[0.1, 0.2, 0.7]`. To determine which prediction the model "made", we'll pick the element with the highest probability. In the case where our model outputs a specific class instead of a vector of probabilities, we can use the same one-hot encoding strategy as we do for the ground truth.

It is common for the actual values or classes to be called `ys` and the predicted values to be called `yhats`, and that is the terminology I use in [my code](https://github.com/BrysonL/groupchat-classification/blob/main/classifier_evaluator.py). Here's the implementation of my accuracy function:

```python
def evaluate(self, ys, yhat):
        yhat_one_hot = F.one_hot(yhat.argmax(dim=1), num_classes=ys.shape[1])

        correct_predictions = (ys == yhat_one_hot).all(dim=1).float().sum()
        accuracy = correct_predictions / ys.shape[0]

        return accuracy.item()
```

This code uses pytorch and some specific functions therein, but it follows the approach outlined above: take the predicted and actual values, choose which class the prediction indicated, and then calculate the accuracy as the proportion of the time the model made the right choice.

While less quantitative, we'll also add the ability for our ClassifierEvaluator to create a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). A confusion matrix will tell us how often the model predicted each sender when it should have been another. This can give us insight like which of the senders the model has a hard time distinguishing between. It won't be as quantitatively useful for comparing models (i.e. we'll have a hard time using it to tell if model A is better than model B), but it will be generally helpful at identifying the strengths and weaknesses of each model. (As an aside, it can also be used to calculate the accuracy by summing along the diagonal and dividing by the total number of messages.):

```python
def compute_confusion_matrix(self, true, pred):
    n = true.shape[0]
    # Create an index tensor of shape [n, 2] where the first column is the true class and the second is the predicted class
    indices = torch.stack((true, pred), dim=1)
    
    # Use bincount to count occurrences of each pair of indices
    conf_matrix = torch.bincount(indices[:, 0] * self.num_classes + indices[:, 1], minlength=self.num_classes**2)
    
    # Reshape the resulting vector to get the 2D confusion matrix
    conf_matrix = conf_matrix.reshape(self.num_classes, self.num_classes)
    
    return conf_matrix
```

Testing this code out on a random set off 100 predictions and ground truths as an illustration yields the following matrix:
||   Bob | Fred   |  John  |
|----|----|----|----|
|**Bob**| 10 | 13 | 14 |
|**Fred**|  6 | 11 | 16 |
|**John**|  9 | 12 |  9 |

In confusion matrices, the rows represent the actual values and columns represent the predicted values. Using the above as an example, this means that when the true value was Bob, our random model predicted the value was Bob 10 times, Fred 13 times, and John 14 times. To calculate the accuracy, you can sum along the diagonals, in this case that's (1,1), (2,2) and (3,3). Dividing that by the total number of trials (100) gives us an accuracy of 30/100 or 30%, about what you'd expect for random selection between 3 options.
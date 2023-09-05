---
title: "Group Chat Classificiation Part 2: Model Evaluation and Naive Classifier"
date: 2023-08-23
---
In the [previous blog](../chat-classification-pt1), we loaded in my group chat messages and looked at some of the trends in the data. We saw that the messages were pretty well distributed across senders and notes some potential patterns we may want to consider when building our models.

Before we can start chugging away on models, though, we have to answer the question: How do we know if we built a good model? There are many ways to answer this question. You can go really deep down the rabbit hole of measuring model quality (ask ChatGPT if you're curious), but for the sake of this post we will use the basic measure of [accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy). Accuracy is defined as the proportion of the time you correctly classify the input. While accuracy is useful, it does have its shortfalls - for example, if your data set is heavily skewed to a single class or if there is a significantly different cost for each type of misclassification. But since our data is evenly distributed and each mistake is considered equal we're happy with this metric choice.

In our case, this means the proportion of time that we correctly identify who sent a message. For example, if our model correctly predicted 4 messages' senders and incorrectly predicted 6 messages' senders, our accuracy would be 4 / (4+6) = 40%. 

In this article, we'll write some code to evaluate the accuracy of models we build, construct a basic model, and evaluate that basic model on our message data.

<!-- more -->

### Building a Classifier Evaluator
As mentioned above, accuracy can be calculated as the number of correct predictions divided by the total number of predictions. There are multiple ways we could represent this in code (each potential class could be represented as a string, each class could be coded as an integer, etc.), but we will take the common Machine Learning approach of vectorizing both the predictions and ground truth. The ground truth will be encoded using [one-hot encoding](https://en.wikipedia.org/wiki/One-hot), where each potential class is an element in a vector (list of integers) with all values being 0 except for location corresponding to the correct output, which is set to 1. For example if we have a list of senders Carn, Ender, and Bean and a particular message was sent by Bean, the one hot vector may look like: `[0, 0, 1]` where the positions in the vector represent `[Carn, Ender, Bean]`. 

Vectorizing the model output will be simple since most of the models we work with in this series will be probabilistic models. Probabilistic models output probabilities for each class given some input, so their output will usually already be in vector format. In the example above, if we had a good model it might output `[0.1, 0.2, 0.7]`. To determine which prediction the model made, we'll pick the element with the highest probability. 

In the case where our model outputs a specific class instead of a vector of probabilities, we can use the same one-hot encoding strategy as we do for the ground truth.

For historical reasons (ask ChatGPT), it is common for the actual values or classes to be called `ys` and the predicted values to be called `yhats`, and that is the terminology I use in [my code](https://github.com/BrysonL/groupchat-classification/blob/main/classifier_evaluator.py). Here's the implementation of my accuracy function:

```python
def evaluate(self, ys, yhat):
        yhat_one_hot = F.one_hot(yhat.argmax(dim=1), num_classes=ys.shape[1])

        correct_predictions = (ys == yhat_one_hot).all(dim=1).float().sum()
        accuracy = correct_predictions / ys.shape[0]

        return accuracy.item()
```

This code uses [PyTorch](https://pytorch.org/docs/stable/index.html) and some specific functions therein, but it follows the approach outlined above: take the predicted and actual values, choose which class the prediction indicated, and then calculate the accuracy as the proportion of the time the model made the right choice.

While less quantitative, we'll also add the ability for our ClassifierEvaluator to create a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix). A confusion matrix will compare the actual and predicted values across all senders rather than the binary right/wrong like accuracy. This can give us insight like which of the senders the model has a hard time distinguishing between. It won't be as quantitatively useful for comparing models (i.e. we'll have a hard time using it to tell if model A is better than model B), but it will be generally helpful at identifying the strengths and weaknesses of each model. (As an aside, it can also be used to calculate the accuracy by summing along the diagonal and dividing by the total number of messages.)

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
<div class="extra-space">

||   Carn | Ender   |  Bean  |
|----|----|----|----|
|**Carn**| 10 | 13 | 14 |
|**Ender**|  6 | 11 | 16 |
|**Bean**|  9 | 12 |  9 |

</div>

In confusion matrices, the rows represent the actual values and columns represent the predicted values. Using the above as an example, this means that when the true value was Carn, our random model predicted the value was Carn 10 times, Ender 13 times, and Bean 14 times. To calculate the accuracy, you can sum along the diagonals; in this case that's (1,1), (2,2) and (3,3). Dividing that by the total number of trials (100) gives us an accuracy of 30/100 or 30%, about what you'd expect for random selection between 3 options. (Though real life is more random than you'd expect, I had to run a handful of times before getting something that looked believable!)

You can find the full [ClassifierEvaluator](https://github.com/BrysonL/groupchat-classification/blob/main/classifier_evaluator.py) code on Github.

### Building a basic model
Now that we've built and tested the code for evaluating models, we can build our first predictive model. I'm going to start with a basic Model class that we can extend for all of our future models. If we were exclusively using one type of deep learning (like Neural Nets), we would probably want to use or extended common classes (like [PyTorch's Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)) for our model. Since we'll be using a variety of techniques and I want to make sure we understand the material, we'll roll our own classes for the most part. Here's the base model we'll extend:

```python
class BaseModel:
    def __init__(self):
        self.model_type = None
        self.trained = False
        # We'll add more variables later as we identify commonalities between models

    def train_model(self, train_data, train_labels, **kwargs):
        raise NotImplementedError("train_model method must be implemented by subclasses.")

    def predict(self, data):
        raise NotImplementedError("predict method must be implemented by subclasses.")
```

We're starting with only `train_model` and `predict` functions, but we'll expand that list as needed as we learn. For example, we'll likely want `save` and `load` functions when models get more complex so we don't have to train from scratch every time we evaluate a model.

In classification, a very simple classification model (hereafter called a classifier) is often called a "naive classifier". The purpose of these models is to have a baseline to which we can compare future models. There are a number of common ways to define a naive classifier, but the implementation we will go with here is called a "most frequent class classifier". As the name suggests, this model will simply pick the class most frequently seen in the training data. When compared to future models, this lets us understand "How much better is this new model than if we simply guessed the most active sender every time?" To illustrate how this model works, consider an example: if we have 100 messages from Carn, Bean, and Ender with a breakdown of 45 from Carn, 35 from Bean, and 20 from Ender, our simple model would always predict Carn no matter what content was passed to the model. Let's take a look at the implementation (with some comments and basic code removed for space reasons):

```python
class MostFrequentClassClassifier(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model_type = "MostFrequentClass"
        self.most_frequent_class = None
        self.num_classes = num_classes
    
    def train_model(self, train_data, train_labels, **kwargs):
        # Convert one-hot encoded labels back to class indices
        label_indices = torch.argmax(train_labels, dim=1)
        
        # Identify the most frequent class
        unique, counts = label_indices.unique(return_counts=True)
        self.most_frequent_class = unique[torch.argmax(counts)]
        
        self.trained = True

    def predict(self, data):
        # Return one-hot encoded predictions
        predictions = torch.zeros(len(data), self.num_classes)
        predictions[:, self.most_frequent_class] = 1
        return predictions
```

When we train the model, we use `argmax` to find the index of the class corresponding to the "one-hot" label, count up the number of times each index (aka sender) appeared, and then select the index that has appeared the most frequently. When we predict, we simply return a one-hot vector with the most frequent class set to 1. If you're confused about the specific pytorch functions used, ask ChatGPT.

You can find the full [BaseModel](https://github.com/BrysonL/groupchat-classification/blob/main/models/model.py) and [MostFrequentClassClassifier](https://github.com/BrysonL/groupchat-classification/blob/main/models/most_frequent_classifier.py) code on Github.

### Training and evaluating the naive model
With the model code complete, we can now turn to training and evaluating the model. In statistics and machine learning, you typically split the data into training and testing data. The training data is used to train your model (shocker) and the testing data is used to measure the model's accuracy. You do this to better estimate how the model will perform on data it hasn't seen before. This helps identify [model overfit](https://en.wikipedia.org/wiki/Overfitting) - where your model gets good at predicting your training data but adapts too much to spurious patterns in that data and then won't work well with new data. We'll use a 90-10 split of training-testing data (as we progress we'll break that 90 down even more for tuning model hyper-parameters). We'll also use a consistent seed (something used by random number generators to make random numbers) to ensure we can replicate results when running the code multiple times.

When we train and evaluate the naive MostFrequentClassClassifier on the model, we see that it has **an accuracy of 27.97%**, which matches closely with the most frequent sender's frequency in our dataset of 27.87%. To examine further, let's look at the confusion matrix:

<div class="extra-space">

|       | Ender | Bean | Alai | Dink | Carn |
|-------|-------|------|------|------|------|
| **Ender** | 0     | 1613 | 0    | 0    | 0    |
| **Bean**  | 0     | 3138 | 0    | 0    | 0    |
| **Alai**  | 0     | 2896 | 0    | 0    | 0    |
| **Dink**  | 0     | 1311 | 0    | 0    | 0    |
| **Carn**  | 0     | 2260 | 0    | 0    | 0    |

</div>

Notice that the model only predicted values of Bean as represented by numbers only being in Bean's column of the confusion matrix. This implies that Bean was the most common sender in our training dataset. Again, we can calculate the accuracy by summing along the diagonal and dividing by the total number gives us 3138/11218 or 27.97%.

You can see the full code for [splitting the data](https://github.com/BrysonL/groupchat-classification/blob/7fe5a29755e9d2a749e38689cd37f4738e3873e0/data_load.py#L44C81-L44C81) and [evaluating the model](https://github.com/BrysonL/groupchat-classification/blob/main/test_files/test_linear_model.py) on Github.

### Conclusion
We now have extensible model and evaluator classes that we can build off of in future posts. The model base and classifier will be useful as we progress in order to build and compare models, respectively. We also used a naive most frequent classifier to establish a baseline of performance we can compare future models to. 

To start building more complex models, we'll need to identify components or properties (aka "features") of our messages that we can use as predictors of who sent each message. Stay tuned for the next blog which will focus on extracting these features from our messages!
---
title: "Group Chat Classificiation Part 4: Linear Models"
date: 2023-08-28
draft: True
---

Welcome to part 4 in my classification blog series! In [part 1](../chat-classification-pt1), we loaded a decade's worth of group chat messages into Python, in [part 2](../chat-classification-pt2) we learned how to quantify model accuracy and tested that knowledge with a basic model, and in [part 2](../chat-classification-pt3) we dug into our data to extract the features we'll need for our models.

Today, we're going to build a model one level up from the naive model in part 2 - a linear model.

Disclaimer that gets more relevant with each post: I'm not a math major nor someone deeply steeped in the theory, so I've been and will continue to keep my explanations in this series high level. If you want to learn more about anything in this series, YouTube and ChatGPT are your friends. There are likely many things I'm wrong about in this series, so if you see something please let me know.

<!-- more -->

### Background and intuitive explanation
*If you already understand linear models or don't care about theory you can skip down to the next section.*

Linear models are a way to represent the relationship between an input set of features and the output variable of concern. They are called "linear" models because they represent the relationship between the variables using a linear equation. Linear equations are the kind you encountered in third grade: `y = m*x + b`. In the language we've been using, that translates to: y (outcome) = m (weight) * x (feature) + b (intercept/bias) where weight corresponds to the value you multiply each feature by and the interecept is the value assigned when all features = 0 (though the interpretation of what that means can get very complicated).

When our outcome variable is continuous (numerical and able to take on any value), the linear model is called a linear regression. Visualized in two dimensions (x and y, or one feature and the outcome), linear regressions look like this:

{{< img src="img/blog/classification/class-linearmodel.jpeg" alt="Basic linear regression model" class="container tall-img">}}

In the visualization above, the red line is the predicted values of y (aka `yhat` from the ClassifierEvaluator in part 2). The model is optimized so that the difference between the red line and the actual points is minimized. The process is the same if there is more than one x, it is just harder to visualize. Given your input of x's (features), assign weights to each x such that you minimize the difference between the actual and predicted y values. The interpretation of these models is also pretty simple - for each one unit increase or change in x, the outcome variable y will change by the weight amount. Figuring out how to set the weights is complicated, but smart math people have made smart math equations to do that. We won't go into the math here.

When you change from a continuous outcome variable to a discrete one, though, the math and intuitive understanding gets harder. We'll start with a binary outcome variable (one that can take on one of two values), which is called a "logistic regression model":

{{< img src="img/blog/classification/class-logisticmodel.jpeg" alt="The challenge of logistic regression" class="container tall-img">}}

As you can see, this is a little more complicated to understand and explain. If the outcome variable can only be two values, "yes" or "no", what does the red line mean at all points that aren't "yes" or "no"? We explain this by using the model to calculate a probability that the outcome is "yes" or "no" rather than the value "yes" or "no" itself. We still essentially minimize the difference between the actual values "yes" or "no" and the predicted probability of "yes" or "no" occurring. We interpret the weights in this model in this way - for each one unit increase or change in x, the outcome y is \<weight> more likely to occur. The smart math people again come through and have nice equations to calculate the weights.

It gets more complicated again when we move to multiple classes:

{{< img src="img/blog/classification/class-multiclassmodel.jpeg" alt="The challenge of logistic regression" class="container tall-img">}}

There are a number of ways to think about this problem which I won't go into here. For our purposes, it is good enough as thinking of our multi-class linear model as producing a set of probabilities for class and then picking the class with the highest probability. For example, if we have three potential senders Ender, Bean, and Carn, our model would output three values: the probability that Ender, Bean, or Carn is the sender of the message. Unfortunately the smart math people have not come through for us here and there is no "right" way to calculate the weights. There are a variety of acceptable options and I'll explain the path we take in our code below.

### Building a linear classifier
Hopefully the above gave you an intuitive grasp of how a linear model works. Now we turn our attention to extending the Model class from part 2 to build our own linear model for classifying messages. We'll pick one of the many ways you can construct a linear multi-class classification model for this exercise: a [one-vs-all model](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/one-vs-all). In this model architecture, we compute one binary model for each of the classes we have, and then choose the maximum value output from those sub models as the predicted class.

Using our messages as an example, if we're trying to predict whether Ender, Bean, or Carn sent a message we would train three binary models: one signifying "did Ender send or did someone besides Ender send?," another for "did Bean send or did not Bean send?," and a third for Carn. When predicting the sender for a new message, we'd run the input through all three models and then pick the max. You can see the full code on the Github, but I'll explain the training and prediction code here. I started off building everything from scratch for you all, but after a couple of hours fighting with optimization code I caved and decided to use the scikit learn builtin logistic regression. I should have given the smart math people more credit above, it's harder than it looks. So the extent of the math you get me to comment on for setting weights is "use a package with it builtin."

Here is how we train each of our binary models (with some filler removed for space reasons, full code on Github):

```python
from sklearn.linear_model import LogisticRegression

def _binary_train(self, train_data, train_labels):
    # Train the model
    model = LogisticRegression(max_iter=1000, C=100.0)
    model.fit(train_data, train_labels)
    return model

def train_model(self, train_data, train_labels, **kwargs):
    # Compute and store the normalization parameters
    self.feature_means = train_data.mean(axis=0)
    self.feature_stds = train_data.std(axis=0)
    
    # Normalize the training data
    train_data = (train_data - self.feature_means) / self.feature_stds

    for i in range(self.num_classes):
        # For each class, set labels to 1 for that class and 0 for all others
        print(f"Training binary classifier for class {i}")
        binary_labels = train_labels[:, i].numpy()
        self.models[i] = self._binary_train(train_data, binary_labels)
    self.trained = True
```

To train the model, we train 5 (aka `self.num_classes`) individual logistic regression models. Many models are sensitive to the scale of variables due to some math reasons, so we also normalize the values using the [standard score](https://en.wikipedia.org/wiki/Standard_score) normalization. This makes sure that all of the features we use are of roughly the same magnitude. Not pictured here, I also updated the feature extraction code so that our categorical variables like previous sender and day of week are now coded in a bunch of boolean variables so that our model can handle them properly. This is similar to the one-vs-all method we used - for the 7 days of the week, I made 7 boolean variables for isMonday, isTuesday, etc. and added them to the feature vector in place of the day of week feature.

When we predict out the data, we have to use the same normalization:
```python
def predict(self, data):
    # Normalize data using the stored parameters
    data = (data - self.feature_means) / self.feature_stds
    
    # Compute scores for each class using the trained models
    scores = [model.decision_function(data) for model in self.models]
    
    # Convert scores to torch tensor and transpose
    return torch.tensor(scores).T
```

To apply the model, we pass the features for the data to the model. When we get to the fun deep learning models that was the point of this blog series, we'll go more in depth on applying the models. Note here that when we return the scores calculated by the model, we transpose the data from dimensions [`num_classes`, `len_data`] (e.g. 5 rows of 10k data points) to dimensions [`len(data)`, `num_classes`] (e.g. 10k rows of the 5 values from our model).

You can find the full linear model code up on Github.

### Testing the linear classifier
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

When you change from a continuous outcome variable to a discrete one, though, the math and intuitive understanding gets harder. We'll start with a binary outcome variable (one that can take on one of two values):

{{< img src="img/blog/classification/class-logisticmodel.jpeg" alt="The challenge of logistic regression" class="container tall-img">}}

As you can see, this is a little more complicated to understand and explain. If the outcome variable can only be two values, "yes" or "no", what does the red line mean at all points that aren't "yes" or "no"? We explain this by using the model to calculate a probability that the outcome is "yes" or "no" rather than the value "yes" or "no" itself. We still essentially minimize the difference between the actual values "yes" or "no" and the predicted probability of "yes" or "no" occurring. We interpret the weights in this model in this way - for each one unit increase or change in x, the outcome y is \<weight> more likely to occur. The smart math people again come through and have nice equations to calculate the weights.

It gets more complicated again when we move to multiple classes:

{{< img src="img/blog/classification/class-multiclassmodel.jpeg" alt="The challenge of logistic regression" class="container tall-img">}}

There are a number of ways to think about this problem which I won't go into here. For our purposes, it is good enough as thinking of our linear model as producing a set of probabilities that each input belongs to each class and then picking the one that seems most likely. For example, if we have three potential senders Ender, Bean, and Carn, our model would output three values: the probability that Ender, Bean, or Carn is the sender of the message. There are a variety of ways to calculate the weights in these problems, and I'll explain the path we take in our code below.

### Building a linear classifier
Hopefully the above gave you an intuitive grasp of how a linear model works. Now we turn our attention to extending the Model class from part 2 to build our own linear model for classifying messages.
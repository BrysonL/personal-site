---
title: "Group Chat Classificiation Part 3: Features"
date: 2023-08-24
draft: True
---

Welcome back to Part 3 in my blog series on classification! In [part 1](../chat-classification-pt1), we loaded a decade's worth of group chat messages into Python and in [part 2](../chat-classification-pt2) we learned how to classify model accuracy and tested that knowledge with a basic model.

Every model from here on out will rely on individual, measurable properties about each message to help predict who sent that message. This information could be anything related to the message that we can measure - what time of day was it sent? how many words did they use? who sent the message directly before this one? In machine learning, these pieces of information are called "features". In later blogs, we may encounter models that can detect their own features or more advanced feature generation methods, but for now this will be similar to the basic data analysis in part 1.

In this blog, we'll examine some techniques for feature identification and highlight some interesting trends we see with potential features in our dataset.

### Identifying features
Often times in classification problems, you will be provided a dataset that already has the features identified, like in the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). In our case, however, we are left to fend for ourselves. Each model we use will likely place its own restrictions and assumptions on the types of features we can use (e.g. the relative scales of each feature, the distribution of the values of a feature across the dataset, the independence of features in the model), but for the purpose of this blog we'll simply try to identify plausible features and transform and modify them as needed later. 

For most models, you should have a clear idea of why you think each feature you choose may impact the outcome you are trying to predict. Even if they appear to be important in your dataset, including features without a clear explanation often ends up with an overfit model. For example, we probably shouldn't use the number of milliseconds since the last second as a feature because we don't have a good hypothesis for why that might impact who sent a message when. In contrast, time of day may be a good feature to include if we think different senders may have different times they tend to be active.

To identify a feature, it is important to not only calculate that feature but also to quantify or visualize that feature with respect to your outcome variable to see if that feature is predictive of your outcome. In future blogs we'll examine more complex modeling and "regularization" techniques that are more robust to unimportant features, but for the first few models we try out we'll want to make sure that each feature we use is chosen for a reason.
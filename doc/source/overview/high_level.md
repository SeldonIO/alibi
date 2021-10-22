# Overview of Explainability

While the applications of machine learning are impressive many models provide predictions that are hard to interpret or
reason about. This limits there use in many cases as often we want to know why and not just what a models prediction 
is. Alarming predictions are rarely taken at face value and typically warrant further analysis. Indeed, in some cases 
explaining the choices that a model makes could even become a potential 
[legal requirement](https://arxiv.org/pdf/1711.00399.pdf).

Explainability research provides us with algorithms that give insights into the context of trained models predictions.

- How does a prediction change dependent on feature inputs?
- What features are Important for a given prediction to hold?
- What features are not important for a given prediction to hold?
- What set of features would you have to minimally change to obtain a new prediction of your choosing?
- etc..

The set of insights available are dependent on the trained model. If the model is a regression is makes sense to ask 
how the prediction varies with respect to some feature whereas it doesn't make sense to ask what minimal change is 
required to obtain a new classification. Insights are constrained by:

- The type of data the model handles (images, text, ...)
- The type of model used (linear regression, neural network, ...)
- The task the model performs (regression, classification, ...)

## Applications:

- At a core level explainability builds trust in the machine learning systems we use. It allows us to justify there use 
in many contexts where an understanding of the basis of decision is paramount.
- Explainability can be thought as an extra form of testing for a model. The insights derived should conform to the
expected behaviour. Failure to do so may indicate issues with the model or problems with the dataset it's been trained 
on.
- Insights can also be used to augment model functionaluty. Providing useful information on top of model predictions. 
How to change the model inputs to obtain a better output for instance.
- Explainability allows researchers to look inside the black box and see what the models are doing. Helping them 
understand more broadly the effects of the particular model or training schema they're using.


## Insights

### Global and Local Insights

Insights can be categorized into two types. Local and global. Intuitively a local insights says something about a 
single prediction that a model makes. As an example, given an image classified as a cat by a model what is the minimal 
set of features (pixels) that need to stay the same in order for that image to still be classified as a cat. Such an 
insight gives an idea of what the model is looking for when deciding to classify an instance into a specific class. 
Global insights on the other hand refer to the behaviour of the model over a set of inputs. Plots that show how a 
regression prediction varies with respect to a given feature while factoring out all the others are an example. These 
insights give a more general understanding of the relationship between inputs and model predictions.

### Insight Categories

Alibi provides a number of insights with which to explore and understand models.

#### Counter Factuals:

Given an instance of the dataset and a prediction given by a model a question that naturally arises is how would the
instance minimally have to change in order for a different prediction to be given. Counterfactuals are local 
explanations as they relate to a single instance and model prediction.

Given a classification model trained
on MNIST and a sample from the dataset with a given prediction, a counter factual would be a generated image that
closely resembles the original but is changed enough that the model correctly classifies it as a different number.

Similarly, given tabular data that a model uses to make financial decisions about a customer a counter factual would
explain to a user how to change they're behaviour in order to obtain a different decision. Alternatively it may tell
the Machine Learning Engineer that the model is drawing incorrect assumptions if the recommended changes involve
features that shouldn't be relevant to the given decision. This may be down either to the model training or the dataset
being unbalanced.

#### Anchors:

Given a single instance and model prediction anchors are local explanations that tell us what minimal set of features 
needs to stay the same in order that the model still give the same prediction or close predictions. This tells the 
practitioner what it is in an instance that most influences the result.

In the case of a trained image classification model an anchor for a given instance and classification would be a 
minimal subset of the image that the model uses to make its decision. A Machine learning engineer might use this 
insight to see if the model is concentrating on the correct image features in making a decision. This is especially 
useful applied to an erroneous decision.

#### Accumulated Local Effect Plot

An ALE-plot shows the dependency of model output on a subset of the input features. This is a global insight as it 
describes the behaviour of the model over the entire input space. This is commonly used to obtain a plot that 
visualizes the relationship directly.

Suppose a trained regression model that predicts the number of bikes rented on a given day dependent on the temperature,
humidity and wind speed. An ALE-plot for the temperature feature is a line graph with temperature plotted against 
number of bikes rented. This type of insight can be used to confirm what you expect to see. In the bikes rented case 
one would anticipate an increase in rentals up until a certain temperature and then a decrease after.

#### Integrated gradients

...
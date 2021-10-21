# Overview of Explainability

While the applications of machine learning are impressive many models provide predictions that are hard to interpret or reason about. This limits there use in many cases as often we want to know why and not just what a models prediction is. Alarming predictions are rarely taken at face value and typically warrant further analysis. Indeed in some cases explaining the choices that a model makes could even become a potential [legal requirement](https://arxiv.org/pdf/1711.00399.pdf).

Explainability research provides us with algorithms that give insights into the context of trained models predictions.

- How does a prediction change dependent on feature inputs?
- What features are Important for a given prediction to hold?
- What features are not important for a given prediction to hold?
- What set of features would you have to minimally change to obtain a new prediction of your choosing?
- etc..

The set of insights available are dependent on the trained model. If the model is a regression is makes sense to ask how the prediction varies with respect to some feature whereas it doesn't make sense to ask what minimal change is required to obtain a new classification. Insights are constrained by:

- The type of data the model handles (images, text, ...)
- The type of model used (linear regression, neural network, ...)
- The task the model performs (regression, classification, ...)

Explainability can be thought as an extra form of testing for a model. The insights derived should conform to the expected behaviour. Failure to do so may indicate issues with the model or problems with the dataset it's been trained on. More so explainability insights can also provide useful information on top of model predictions. How to change the model inputs to obtain a better output for instance.


## Insights

...

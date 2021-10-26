# Practical Overview of Explainability

While the applications of machine learning are impressive many models provide predictions that are hard to interpret or
reason about. This limits there use in many cases as often we want to know why and not just what a models prediction 
is. Alarming predictions are rarely taken at face value and typically warrant further analysis. Indeed, in some cases 
explaining the choices that a model makes could even become a potential 
[legal requirement](https://arxiv.org/pdf/1711.00399.pdf). The following is a non-rigorous and practical overview of 
explainability and the methods alibi provide.

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

In particular some explainer methods apply to any type of model. They can do so because the underlying method doesn't
make use of the model internals. Instead, only depending on the model outputs given particular inputs. Methods that 
apply in this general setting are known as **black box** methods. Methods that do require model internals, perhaps in
order to compute prediction gradients dependent on inputs, are known as white box models. This is a much stronger 
constrain that black box methods.

:::{admonition} **Note 1: Black Box Definition**
The use of Black Box here varies subtly from the conventional use within machine learning. Typically, we say a model is
a black box if the mechanism by which it makes predictions is too complicated to be interpretable to a human. Here we
use black box to mean that the explainer method doesn't need to have access to the model internals in order to be 
applied.
:::


## Applications:

**Trust:**

At a core level explainability builds trust in the machine learning systems we use. It allows us to justify there use 
in many contexts where an understanding of the basis of decision is paramount.

**Testing:** 

Explainability can be thought as an extra form of testing for a model. The insights derived should conform to the
expected behaviour. Failure to do so may indicate issues with the model or problems with the dataset it's been trained 
on.

**Functionality:**

Insights can also be used to augment model functionality. Providing useful information on top of model predictions. 
How to change the model inputs to obtain a better output for instance.

**Research:**

Explainability allows researchers to look inside the black box and see what the models are doing. Helping them 
understand more broadly the effects of the particular model or training schema they're using.

:::{admonition} **Note 2: Biases**
Practitioners must be wary of using explainability to excuse bad models rather than ensuring there correctness. As an 
example its possible to have a model that is correctly trained on a dataset, however due to the dataset being either 
wrong or incomplete the model doesn't actually reflect reality. If the insights that explainability generates 
conform to some confirmation bias of the person training the model then they are going to be blind to this issue and
instead use these methods to confirm erroneous results.
:::

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

A counterfactual, $x_{cf}$, needs to satisfy

- The model prediction on $x_{cf}$ needs to be close to the predefined output.
- The counterfactual $x_{cf}$ should be interpretable. 

The first requirement is easy enough to satisfy. The second however requires some idea of what interpretable means. In
our case we require that the perturbation $\delta$ changing the original instance $x_0$ into $x_{cf} = x_0 + \delta$ 
should be sparse. This means we prefer solutions that change a small subset of the features to construct $x_{cf}$. This
is limits the complexity of the solution making it more understandable. Secondly we want $x_{cf}$ to lie close to both 
the overall data distribution and the counterfactual class specific data distribution. This condition ensures the
counter factual makes sense as something that would both occur in the dataset and occur within the target counter
factual class.

##### Explainers:

The following discusses the set of explainer methods available from alibi for generating counterfactual insights.

**Counterfactuals Instances:**

TODO

**Contrastive Explanation Method:**

TODO

**Counterfactuals Guided by Prototypes:**

TODO

**Counterfactuals with Reinforcement Learning:**

TODO

___

#### Local Scoped Rules (Anchors):

Given a single instance and model prediction anchors are local explanations that tell us what minimal set of features 
needs to stay the same in order that the model still give the same prediction or close predictions. This tells the 
practitioner what it is in an instance that most influences the result.

In the case of a trained image classification model an anchor for a given instance and classification would be a 
minimal subset of the image that the model uses to make its decision. A Machine learning engineer might use this 
insight to see if the model is concentrating on the correct image features in making a decision. This is especially 
useful applied to an erroneous decision.

We introduce anchors in a more formal manner, taking the definition and discussion from *.

Let A be a rule (set of predicates) acting on such an interpretable representation, such that $A(x)$ returns $1$ if all 
its feature predicates are true for instance $x$. An example of such a rule, $A$, could be represented by the set 
$\{not, bad\}$ in which case any sentence, $s$, with both $not$ and $bad$ in it would mean $A(s)=1$

Given a classifier $f$, instance $x$ and data distribution $\mathcal{D}$, $A$ is an anchor for $x$ if $A(x) = 1$ and,

$$ E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}] ≥ τ $$

The distribution $\mathcal{D}(z|A)$ is those points from the dataset for which the anchor holds. This is like fixing 
some set of features of an instance and allowing all the others to vary. Intuitively, the anchor condition says any
point in the data distribution that satisfies the anchor $A$ is expected to match the model prediction $f(x)$ with 
probability $\tau$ (usually $\tau$ is chosen to be 0.95). 

Let $prec(A) = E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}]$ be the precision of an anchor. Note that the precision of an anchor 
is considered with respect to the set of points in the data distribution to which the anchor applies,
$\mathcal{D}(z|A)$. We can consider the **coverage** of an anchor as the probability that $A(z)=1$ for any instance 
$z$ in the data distribution. The coverage tells us the proportion of the distribution that the anchor applies to. 
The aim here is to find the anchor that applies to the largest set of instances. So what is the most general rule we 
can find that any instance must satisfy in order that it have the same classification as $x$.

##### Explainers:

The following discusses the set of explainer methods available from alibi for generating anchor insights.

**Anchors**

TODO

**Contrastive Explanation Method:**

TODO


#### Global Feature Attribution

Global Feature Attribution methods aim to show the dependency of model output on a subset of the input features. This 
is a global insight as it describes the behaviour of the model over the entire input space. ALE-plots, M-plots and 
PDP-plots all are used to obtain graphs that visualize the relationship between feature and prediction directly.

Suppose a trained regression model that predicts the number of bikes rented on a given day dependent on the temperature,
humidity and wind speed. Global Feature Attribution for the temperature feature might be a line graph with temperature 
plotted against number of bikes rented. This type of insight can be used to confirm what you expect to see. In the 
bikes rented case one would anticipate an increase in rentals up until a certain temperature and then a decrease after.

**Accumulated Local Effects:**

Alibi only provides Accumulated Local Effects plots because these give the most accurate insight of ALE-plots, M-plots 
and PDP-plots. ALE-plots work by averaging the local changes in prediction at every instance (local effects) in the 
data distribution. They then accumulate these differences to obtain a plot of prediction over the selected feature 
dependencies.

Suppose we have a model $f$ and features $X={x_1,... x_n}$. Given a subset of the features $X_S$ then let 
$X_C=X \\ X_S$. We want to obtain the ALE-plot for the features $X_S$ typical chosen to be at most 2 in order that 
they can easily be visualized. The ALE-plot is defined as:

$$
\hat{f}_{S, ALE}(X_S) = 
\int_{z_{0, S}}^{S} \mathbb{E}_{X_C|X_S=x_s} 
\left[ \frac{\partial \hat{f}}{\partial x_S} (X_s, X_c)|X_S=z_S \right]  dz_{S} - constant
$$

TODO: further discussion on definition

#### Local Feature Attribution

Local feature attribution asks how each feature in a given instance contributes to its prediction. In the case of an 
image this would highlight those pixels that make the model give the output it does. Note this differs subtly from 
local scoped rules in that they find the minimum subset of features required to give a prediction whereas local feature 
attribution creates a heat map that tells us each features contribution to the overall outcome.

A common issue with some attribution methods is that you must measure the contribution with respect to some baseline 
point. The choice of baseline should capture a blank state in which the model makes essentially no prediction or assigns 
probability of classes equally. A common choice for image classification is an image set to black. This works well in 
many cases but sometimes fails to be a good choice. For instance for a model that classifies images taken at night 
using an image with every pixel set to black means the attribution method with undervalue the use of dark pixels in 
attributing the contribution of each feature to the classification. This is due to the contribution being calculated
relative to the baseline which in this case is already dark.

A good use of local feature attribution is to detect that a classifier trained on images is focusing on the correct 
features of an image in order to infer the class. As an example suppose you have a model trained to detect breeds of 
dog. You want to check that it focuses on the correct features of the dog in making its prediction. If you compute
the feature attribution of a picture of a husky and discover that the model is only focusing on the snowy backdrop to
the husky then you know both that all the images of huskies in your dataset overwhelmingly have snowy backdrops and 
also that the model will fail to generalize. It will potentially incorrectly classify other breeds of dog with snowy 
backdrops and also fail to recognise huskies without snowy backdrops.

TODO: discussion on definition

##### Explainers:

The following discusses the set of explainer methods available from alibi for generating Local Feature Attribution 
insights.

**Integrated Gradients**

TODO

**Kernel SHAP**

TODO

**Tree SHAP**

TODO

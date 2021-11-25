## Introduction

```{contents}
:depth: 3
:local: true
```

# What is Explainability?

**Explainability provides us with algorithms that give insights into trained models predictions.** It allows us to
answer questions such as:

- How does a prediction **change** dependent on feature inputs?
- What features **are** or **are not** important for a given prediction to hold?
- What set of features would you have to minimally **change** to obtain a **new** prediction of your choosing?
- How does each feature **contribute** to a model's prediction?

```{image} images/exp-aug.png
:align: center
:alt: Model augmented with explainabilty 
```

Alibi provides a set of **algorithms** or **methods** known as **explainers**. Each explainer provides some kind of
insight about a model. The set of insights available given a trained model is dependent on a number of factors. For
instance, if the model is a [regression](https://en.wikipedia.org/wiki/Regression_analysis) it makes sense to ask how
the prediction varies for some regressor. Whereas, it doesn't make sense to ask what minimal change is required to
obtain a new class prediction. In general, given a model the explainers we can use are constrained by:

- The **type of data** the model handles. Each insight applies to some or all of the following kinds of data: image,
  tabular or textual.
- The **task the model** performs, regression
  or [classification](https://en.wikipedia.org/wiki/Statistical_classification).
- The **type of model** used. Examples of model types
  include [neural networks](https://en.wikipedia.org/wiki/Neural_network)
  and [random forests](https://en.wikipedia.org/wiki/Random_forest).

## Applications

As machine learning methods have become more complex and more mainstream, with many industries
now [incorporating AI](https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/global-survey-the-state-of-ai-in-2020)
in some form or another, the need to understand the decisions made by models is only increasing. Explainability has
several applications of importance.

- **Trust:**. At a core level, explainability builds [trust](https://onlinelibrary.wiley.com/doi/abs/10.1002/bdm.542) in
  the machine learning systems we use. It allows us to justify their use in many contexts where an understanding of the
  basis of the decision is paramount. This is a common issue within machine learning in medicine, where acting on a
  model prediction may require expensive or risky procedures to be carried out.
    - **Testing:**. Explainability might be used to [audit financial models](https://arxiv.org/abs/1909.06342) that aid
      decisions about whether to grant customer loans. By computing the attribution of each feature towards the
      prediction the model makes, organisations can check that they are consistent with human decision-making.
      Similarly, explainability applied to a model trained on image data can explicitly show the model's focus when
      making decisions, aiding [debugging](http://proceedings.mlr.press/v70/sundararajan17a.html). Practitioners must be
      wary of [misuse](#biases), however.
- **Functionality:**. Insights can be used to augment model functionality. For instance, providing information on top of
  model predictions such as how to change model inputs to obtain desired outputs.
- **Research:**. Explainability allows researchers to understand how and why opaque models make decisions. This can help
  them understand more broadly the effects of the particular model or training schema they're using.

## Black-box vs White-box methods

Some explainers apply only to specific types of models such as the [Tree SHAP](#path-dependent-treeshap) methods which
can only be used with [tree-based models](https://en.wikipedia.org/wiki/Decision_tree_learning). This is the case when
an explainer uses some aspect of that model's internal structure. If the model is a neural network then some methods
require taking gradients of the model predictions with respect to the inputs. Methods that require access to the model
internals are known as **white-box** methods. Other explainers apply to any type of model. They can do so because the
underlying method doesn't make use of the model internals. Instead, they only need to have access to the model outputs
given particular inputs. Methods that apply in this general setting are known as **black-box** methods. Note that
white-box methods are a subset of black-box methods and an explainer being a white-box method is a much stronger
constraint than it being a black-box method. Typically, white-box methods are faster than black-box methods as they can
exploit the model internals.

:::{admonition} **Note 1: Black-box Definition**
The use of black-box here varies subtly from the conventional use within machine learning. In most other contexts a
model is a black-box if the mechanism by which it makes predictions is too complicated to be interpretable to a human.
Here we use black-box to mean that the explainer method doesn't need access to the model internals to be applied.
:::

## Global and Local Insights

Insights can be categorised into two categories &mdash; Local and global. Intuitively, a local insight says something
about a single prediction that a model makes. For example, given an image classified as a cat by a model, a local
insight might give the set of features (pixels) that need to stay the same for that image to remain classified as a cat.

On the other hand, global insights refer to the behaviour of the model over a range of inputs. As an example, a plot
that shows how a regression prediction varies for a given feature. These insights provide a more general understanding
of the relationship between inputs and model predictions.

```{image} images/local-global.png
:align: center
:alt: Local and Global insights 
```

## Biases

The explanations Alibi's methods provide depend on the model, the data, and &mdash; for local methods &mdash; the
instance of interest. Thus Alibi allows us to obtain insight into the model and, therefore, also the data, albeit
indirectly. There are several pitfalls of which the practitioner must be wary.

Often bias exists in the data we feed machine learning models even when we exclude sensitive factors. Ostensibly
explainability is a solution to this problem as it allows us to understand the model's decisions to check if they're
appropriate. However, human bias itself is still an element. Hence, if the model is doing what we expect it to on biased
data, we are venerable to using explainability to justify relations in the data that may not be accurate. Consider:
> _"Before launching the model, risk analysts are asked to review the Shapley value explanations to ensure that the
> model exhibits expected behavior (i.e., the model uses the same features that a human would for the same task)."_
> &mdash; <cite>[Explainable Machine Learning in Deployment](https://dl.acm.org/doi/pdf/10.1145/3351095.3375624)</cite>

The critical point here is that the risk analysts in the above scenario must be aware of their own bias and potential
bias in the dataset. The Shapley value explanations themselves don't remove this source of human error; they just make
the model less opaque.

Machine learning engineers may also have expectations about how the model should be working. An explanation that doesn't
conform to their expectations may prompt them to erroneously decide that the model is "incorrect". People usually expect
classifiers trained on image datasets to use the same structures humans naturally do when identifying the same classes.
However, there is no reason to believe such models should behave the same way we do.

Interpretability of insights can also mislead. Some insights such as [anchors](#anchors) give conditions for a
classifiers prediction. Ideally, the set of these conditions would be small. However, when obtaining anchors close to
decision boundaries, we may get a complex set of conditions to differentiate that instance from near members of a
different class. Because this is harder to understand, one might write the model off as incorrect, while in reality, the
model performs as desired.

# Types of Insights

Alibi provides several local and global insights with which to explore and understand models. The following gives the
practitioner an understanding of which explainers are suitable in which situations.

| Explainer                                                                                    | Scope  | Model types           | Task types                 | Data types                               | Use                                                                                             |
| -------------------------------------------------------------------------------------------- | ------ | --------------------- | -------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [Accumulated Local Effects](#accumulated-local-effects)                                      | Global | Black-box             | Classification, Regression | Tabular                                  | How does model prediction vary with respect to features of interest                             |
| [Anchors](#anchors)                                                                          | Local  | Black-box             | Classification             | Tabular, Categorical, Text and Image     | Which set of features of a given instance is sufficient to ensure the prediction stays the same |
| [Pertinent Positives](#pertinent-positives)                                                  | Local  | Black-box/White-box   | Classification             | Tabular, Image                           | ""                                                                                              |
| [Integrated Gradients](#integrated-gradients)                                                | Local  | White-box             | Classification, Regression | Tabular, Categorical, Text and Image     | What does each feature contribute to the model prediction?                                      |
| [Kernel SHAP](#kernelshap)                                                                   | Local  | Black-box             | Classification, Regression | Tabular, Categorical                     | ""                                                                                              |
| [Tree SHAP (path-dependent)](#path-dependent-treeshap)                                       | Local  | White-box             | Classification, Regression | Tabular, Categorical                     | ""                                                                                              |
| [Tree SHAP (interventional)](#interventional-tree-shap)                                      | Local  | White-box             | Classification, Regression | Tabular, Categorical                     | ""                                                                                              |
| [Counterfactuals Instances](#counterfactuals-instances)                                      | Local  | Black-box/White-box   | Classification             | Tabular, Image                           | What minimal change to features is required to reclassify the current prediction?               |
| [Contrastive Explanation Method](#contrastive-explanation-method)                            | Local  | Black-box/White-box   | Classification             | Tabular, Image                           | ""                                                                                              |
| [Counterfactuals Guided by Prototypes](#counterfactuals-guided-by-prototypes)                | Local  | Black-box/White-box   | Classification             | Tabular, Categorical, Image              | ""                                                                                              |
| [counterfactuals-with-reinforcement-learning](#counterfactuals-with-reinforcement-learning)  | Local  | Black-box             | Classification             | Tabular, Categorical, Image              | ""                                                                                              |

### 1. Global Feature Attribution

Global Feature Attribution methods aim to show the dependency of model output on a subset of the input features. They
are a global insight as it describes the behavior of the model over the entire input space. For instance, Accumulated
Local Effects plots obtain graphs that directly visualize the relationship between feature and prediction.

Suppose a trained regression model that predicts the number of bikes rented on a given day depending on the temperature,
humidity, and wind speed. A global feature attribution plot for the temperature feature might be a line graph plotted
against the number of bikes rented. In the bikes rented case, one would anticipate an increase in rentals up until a
specific temperature and then a decrease after it gets too hot.

### Accumulated Local Effects

| Explainer                    | Scope  | Model types  | Task types                 | Data types  | Use                                                                 |
| ---------------------------- | ------ | ------------ | -------------------------- | ----------- | ------------------------------------------------------------------- |
| Accumulated Local Effects    | Global | Black-box    | Classification, Regression | Tabular     | How does model prediction vary with respect to features of interest |

Alibi only provides accumulated local effects plots because of the available global feature attribution methods they
give the most accurate insight. Alternatives include Partial Dependence Plots. ALE plots work by averaging the local
changes in a prediction at every instance in the data distribution. They then accumulate these differences to obtain a
plot of prediction over the selected feature dependencies.

Suppose we have a model $f$ and features $X=\{x_1,... x_n\}$. Given a subset of the features $X_S$, we denote $X_C=X
\setminus X_S$. We want to obtain the ALE-plot for the features $X_S$, typically chosen to be at most a set of dimension
two to be visualized easily. For simplicity assume we have $X=\{x_1, x_2\}$ and let $X_S=\{x_1\}$ so $X_C=\{x_2\}$. The
ALE of $x_1$ is defined by:

$$ \hat{f}_{S, ALE}(x_1) = \int_{min(x_1)}^{x_1}\mathbb{E}\left[
\frac{\partial f(X_1, X_2)}{\partial X_1} | X_1 = z_1 \right]dz_1 - c_1 $$

The term within the integral computes the expectation of the model derivative in $x_1$ over the random variable $X_2$
conditional on $X_1=z_1$. By taking the expectation for $X_2$, we factor out its dependency. So now we know how the
prediction $f$ changes local to a point $X_1=z_1$ independent of $X_2$. Integrating these changes over $x_1$ from a
minimum value to the value of interest, we obtain the global plot of how the model depends on $x_1$. ALE-plots get their
names as they accumulate (integrate) the local effects (the expected partial derivatives). Note that here we have
assumed $f$ is differentiable. In practice, however, we compute the various quantities above numerically, so this isn't
a requirement.

__TODO__:

- Add picture explaining the above idea.

For more details on accumulated local effects including a discussion on PDP-plots and M-plots
see [Motivation-and-definition for ALE](../methods/ALE.ipynb)

:::{admonition} **Note 4: Categorical Variables and ALE**
Note that because ALE plots require computing differences between variables, they don't naturally extend to categorical
data unless there is a sensible ordering on the data. As an example, consider the months of the year. To be clear, this
is only an issue if the variable you are taking the ALE for is categorical.
:::

**Pros**:

- ALE-plots are easy to visualize and understand intuitively
- Very general as it is a black-box algorithm
- Doesn't struggle with dependencies in the underlying features, unlike PDP plots
- ALE plots are fast

**Cons**:

- Harder to explain the underlying motivation behind the method than PDP plots or M plots.
- Requires access to the training dataset.
- Unlike PDP plots, ALE plots do not work with Categorical data

## 2. Local Necessary Features

Local necessary features tell us what features need to stay the same for a specific instance in order for the model to
give the same classification. In the case of a trained image classification model, local necessary features for a given
instance would be a minimal subset of the image that the model uses to make its decision. Alibi provides two explainers
for computing local necessary features: [anchors](#anchors) and [pertinent positives](#pertinent-positives).

### Anchors

| Explainer  | Scope  | Model types  | Task types      | Data types                           | Use                                                                                             |
| ---------- | ------ | ------------ | --------------- | ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Anchors    | Local  | Black-box    | Classification  | Tabular, Categorical, Text and Image | Which set of features of a given instance is sufficient to ensure the prediction stays the same |

Anchors are introduced
in [Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf). Further,
more detailed documentation can be found [here](../methods/Anchors.ipynb).

Let A be a rule (set of predicates) acting on input instances, such that $A(x)$ returns $1$ if all its feature
predicates are true. Consider the [wine quality dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) adjusted
by partitioning the data into good and bad wine based on a quality threshold of 0.5.

```{image} images/wine-quality-ds.png
:align: center
:alt: first five rows of wine quality dataset 
```

An example of a predicate for this dataset would be a rule of the form: `'alcohol > 11.00'`. Note that the more
predicates we add to an anchor, the smaller it becomes, as by doing so, we filter out more instances of the data.
Anchors are sets of predicates associated to a specific instance $x$ such that $x$ is in the anchor ($A(x)=1$) and any
other point in the anchor has the same classification as $x$ ($z$ such that $A(z) = 1 \implies f(z) = f(x)$ where $f$ is
the model). We're interested in finding the largest possible Anchor that contains $x$.

```{image} images/anchor.png
:align: center
:alt: Illustration of an anchor as a subset of two dimensional feature space. 
```

To construct an anchor using Alibi for tabular data such as the wine quality dataset, we use:

```ipython3
from alibi.explainers import AnchorTabular

predict_fn = lambda x: model.predict(scaler.transform(x))
explainer = AnchorTabular(predict_fn, features)
explainer.fit(X_train, disc_perc=(25, 50, 75))
result = explainer.explain(x, threshold=0.95)

print('Anchor =', result.data['anchor'])
print('Coverage = ', result.data['coverage'])
```

where `x` is an instance of the dataset classified as good.

```ipython3
Anchor = ['sulphates <= 0.55', 'volatile acidity > 0.52', 'alcohol <= 11.00', 'pH > 3.40']
Coverage =  0.0316930775646372
```

Note Alibi also gives an idea of the size (coverage) of the Anchor.

To find anchors Alibi sequentially builds them by generating a set of candidates from an initial anchor candidate,
picking the best candidate of that set and then using that to generate the next set of candidates and repeating.
Candidates are favoured on the basis of the number of instances they contain that are in the same class as $x$ under
$f$. This is repeated until we obtain a candidate that satisfies the condition and is largest (in the case where there
are multiple).

To compute which of two anchors is better, Alibi obtains an estimate by sampling from $\mathcal{D}(z|A)$ where
$\mathcal{D}$ is the data distribution. The sampling process is dependent on the type of data. For tabular data, this
process is easy; we can fix the values in the Anchor and replace the rest with values from a point sampled from the
dataset.

In the case of textual data, anchors are sets of words that the sentence must include to be **in the** anchor. To sample
from $\mathcal{D}(z|A)$, we need to find realistic sentences that include those words. To help do this Alibi provides
support for three [transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) based language
models: `DistilbertBaseUncased`, `BertBaseUncased`, and `RobertaBase`.

Image data being high dimensional means we first need to reduce it to a lower dimension. We can do this using image
segmentation algorithms (Alibi supports
[felzenszwalb](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#felzenszwalb-s-efficient-graph-based-segmentation)
,
[slic](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#slic-k-means-based-image-segmentation)
and [quickshift](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#quickshift-image-segmentation))
to find super-pixels. As a result, the anchors are made up of sets of these super-pixels, and so to sample from
$\mathcal{D}(z|A)$ we replace those super-pixels that aren't in $A$ with something else. Alibi supports superimposing
over the absent super-pixels with an image sampled from the dataset or taking the average value of the super-pixel.

The fact that the method requires perturbing and comparing anchors at each stage leads to some issues. For instance, the
more features, the more candidate anchors you can obtain at each process stage. The algorithm uses
a [Beam search](https://en.wikipedia.org/wiki/Beam_search) among the candidate anchors and solves for the best $B$
anchors at each stage in the process by framing the problem as
a [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit). The runtime complexity is $\mathcal{O}(B \cdot
p^2 + p^2 \cdot \mathcal{O}_{MAB[B \cdot p, B]})$ where $p$ is the number of features and $\mathcal{O}_
{MAB[B \cdot p, B]}$ is the runtime for the multi-armed bandit. (
See [Molnar](https://christophm.github.io/interpretable-ml-book/anchors.html#complexity-and-runtime) for more details.)

Similarly, comparing anchors that are close to decision boundaries can require many samples to obtain a clear winner
between the two. Also, note that anchors close to decision boundaries are likely to have many predicates to ensure the
required predictive property. This makes them less interpretable.

| Pros                                                                                                           | Cons                                                                                                   |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Easy to explain as rules are simple to interpret                                                               | Time complexity scales as a function of features                                                       |   
| Is a black-box method as we need to predict the value of an instance and don't need to access model internals  | Requires a large number of samples to distinguish anchors close to decision boundaries                 |
| The coverage of an anchor gives a level of global insight as well                                              | Anchors close to decision boundaries are less likely to be interpretable                               |
|                                                                                                                | High dimensional feature spaces such as images need to be reduced to improve the runtime complexity    |
|                                                                                                                | Practitioners may need domain-specific knowledge to correctly sample from the conditional probability  |

### Pertinent Positives

| Explainer           | Scope  | Model types          | Task types      | Data types      | Use                                                                                             |
| ------------------- | ------ | -------------------- | --------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| Pertinent Positives | Local  | Black-box/White-box  | Classification  | Tabular, Image  | Which set of features of a given instance is sufficient to ensure the prediction stays the same |

Introduced by [Amit Dhurandhar, et al](https://arxiv.org/abs/1802.07623), a Pertinent Positive is the subset of features
of an instance that still obtains the same classification as that instance. These differ from [anchors](#anchors)
primarily in the fact that they aren't constructed to maximize coverage. The method to create them is also substantially
different. The rough idea is to define an **absence of a feature** and then perturb the instance to take away as much
information as possible while still retaining the original classification. Note that these are a subset of
the [CEM](../methods/CEM.ipynb) method which is also used to
construct [pertinent negatives/counterfactuals](#counterfactuals).

```{image} images/pp_mnist.png
:align: center
:alt: Pertinent postive of an MNIST digit 
```

Given an instance $x$ we use gradient descent to find a $\delta$ that minimizes the following loss:

$$ L = c\cdot L_{pred}(\delta) + \beta L_{1}(\delta, x) + L_{2}^{2}(\delta, x) + \gamma \|\delta - AE(\delta)\|^{2}_{2}
$$

$AE$ is an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) generated from the training data. If $\delta$ strays
from the original data distribution, the autoencoder loss will increase as it will no longer reconstruct $\delta$ well.
Thus, we ensure that $\delta$ remains close to the original dataset distribution.

Note that $\delta$ is constrained to only "take away" features from the instance $x$. There is a slightly subtle point
here: removing features from an instance requires correctly defining non-informative feature values. For
the [MNIST digits](http://yann.lecun.com/exdb/mnist/), it's reasonable to assume that the black background behind each
digit represents an absence of information. Similarly, in the case of color images, you might take the median pixel
value to convey no information, and moving away from this value adds information. For numeric tabular data we can use
the feature mean. In general, having to choose a non-informative value for each feature is non-trivial and domain
knowledge is required.

Note that we need to compute the loss gradient through the model. If we have access to the internals, we can do this
directly. Otherwise, we need to use numerical differentiation at a high computational cost due to the extra model calls
we need to make. This does however mean we can use this method for a wide range of black-box models but not all. We
require the model to be differentiable which isn't always true. For instance tree-based models have piece-wise constant
output.

| Pros                                                                                                           | Cons                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Can be used with both white-box (TensorFlow) and some black-box models                                         | Finding non-informative feature values to take away from an instance is often not trivial, and domain knowledge is essential |   
|                                                                                                                | The autoencoder loss requires access to the original dataset                                                                 |
|                                                                                                                | Need to tune hyperparameters $\beta$ and $\gamma$                                                                            |
|                                                                                                                | The insight doesn't tell us anything about the coverage of the pertinent positive                                            |
|                                                                                                                | Slow for black-box models due to having to numerically evaluate gradients                                                    |
|                                                                                                                | Only works for differentiable black-box models                                                                               |

### Local Feature Attribution

Local feature attribution asks how each feature in a given instance contributes to its prediction. In the case of an
image, this would highlight those pixels that make the model provide the output it does. Note that this differs subtly
from Local Necessary Features, which find the minimum subset of features required to give a prediction. Local feature
attribution instead assigns a score to each feature.

__TODO__:

- picture showing above.

A good example use of local feature attribution is to detect that a classifier trained on images is focusing on the
correct features of an image to infer the class. Suppose you have a model trained to classify breeds of dogs. You want
to check that it focuses on the correct features of the dog in making its prediction. Suppose you compute the feature
attribution of a picture of a husky and discover that the model is only focusing on the snowy backdrop to the husky,
then you know two things. All the images of huskies in your dataset overwhelmingly have snowy backdrops, and also that
the model will fail to generalize. It will potentially incorrectly classify other dog breeds with snowy backdrops as
huskies and fail to recognize huskies that aren't in snowy locations.

Each of the following methods defines local feature attribution slightly differently. In both, however, we assign
attribution values to each feature to indicate how significant those features were in making the model prediction.

Let $f:\mathbb{R}^n \rightarrow \mathbb{R}$. $f$ might be a regression, a single component of a multi regression or a
probability of a class in a classification model. If $x=(x_1,... ,x_n) \in \mathbb{R}^n$ then an attribution of the
prediction at input $x$ is a vector $a=(a_1,... ,a_n) \in \mathbb{R}^n$ where $a_i$ is the contribution of $x_i$ to the
prediction $f(x)$.

The attribution values should satisfy specific properties:

1. Efficiency/Completeness: The sum of attributions equals the difference between the prediction and the
   baseline/average. We're interested in understanding the difference each feature value makes in a prediction compared
   to some uninformative baseline.
2. Symmetry: If the model behaves the same after swapping two variables $x$ and $y$, then $x$ and $y$ have equal
   attribution. If this weren't the case, we would be biasing the attribution towards certain features over other ones.
3. Dummy/Sensitivity: If a variable does not change the output of the model, then it should have attribution 0. If this
   were not the case, we'd be assigning value to a feature that provides no information.
4. Additivity/Linearity: The attribution for a feature $x_i$ of a linear composition of two models $f_1$ and $f_2$ given
   by $c_1 f_1 + c_2 f_2$ is $c_1 a_{1, i} + c_2 a_{2, i}$ where $a_{1, i}$ and $a_{2, i}$ is the attribution for $x_1$
   and $f_1$ and $f_2$ respectively.

### Integrated Gradients

| Model-types | Task-types     | Data-types  |
| ----------- | -------------- | ----------- |
| TF/Kera     | Classification | Tabular     |
|             | Regression     | Image       |
|             |                | Text        |
|             |                | Categorical |

This method computes the attribution of each feature by integrating the model partial derivatives along a path from a
baseline point to the instance. Let $f$ be the model and $x$ the instance of interest. If $f:\mathbb{R}^{n} \rightarrow
\mathbb{R}^{m}$ where $m$ is the number of classes the model predicts then let $F=f_k$ where $k \in \{1,..., m\}$. If
$f$ is single-valued then $F=f$. We also need to choose a baseline value, $x'$.

$$ IG_i(x) = (x_i - x_i')\int_{\alpha}^{1}\frac{\partial F (x' + \alpha (x - x'))}{ \partial x_i } d \alpha $$

The above sums partial derivatives for each feature over the path between the baseline and instance of interest. In
doing so, you accumulate the changes in the prediction that occur due to the changing feature value from the baseline to
the instance.

:::{admonition} **Note 5: Choice of Baseline**
The main difficulty with this method is that as IG is very dependent on the baseline, it's essential to make sure you
choose it well. The choice of baseline should capture a blank state in which the model makes essentially no prediction
or assigns the probability of each class equally. A common choice for image classification is an image set to black,
which works well in many cases but sometimes fails to be a good choice. For instance, a model that classifies images
taken at night using an image with every pixel set to black means the attribution method will undervalue the use of dark
pixels in attributing the contribution of each feature to the classification. This is due to the contribution being
calculated relative to the baseline, which is already dark.
:::

**Pros**

- Simple to understand and visualize, especially with image data
- Doesn't require access to the training data

**Cons**

- white-box method. Requires the partial derivatives of the model outputs with respect to inputs
- Requires choosing the baseline which can have a significant effect on the outcome (See Note 5)

### KernelSHAP

| Model-types       | Task-types     | Data-types  |
| ----------------- | -------------- | ----------- |
| Black-box         | Classification | Tabular     |
|                   | Regression     | Categorical |

Kernel SHAP is a method of computing the Shapley values for a model around an instance $x_i$. Shapley values are a
game-theoretic method of assigning payout to players depending on their contribution to an overall goal. In this case,
the players are the features, and their payout is the model prediction. To compute these values, we have to consider the
marginal contribution of each feature over all the possible coalitions of feature players.

Suppose we have a regression model $f$ that makes predictions based on four features $X = \{X_1, X_2, X_3, X_4\}$ as
input. A coalition is a group of features, say, the first and third features. For this coalition, its value is given by:

$$ val({1,3}) = \int_{\mathbb{R}}\int_{\mathbb{R}} f(x_1, X_2, x_3, X_4)d\mathbb{P}_{X_{2}X_{4}} - \mathbb{E}_{X}(f(X))
$$

Given a coalition, $S$, that doesn't include $x_i$, then the marginal contribution of $x_i$ is given by $val(S \cup x_i)

- val(S)$. Intuitively this is the difference that the feature $x_i$ would contribute if it was to join that coalition.
  We are interested in the marginal contribution of $x_i$ over all possible coalitions with and without $x_i$. A Shapley
  value for the $x_i^{th}$ feature is given by the weighted sum

$$ \psi_j = \sum_{S\subset \{1,...,p\} \setminus \{j\}} \frac{|S|!(p - |S| - 1)!}{p!}(val(S \cup x_i) - val(S))
$$

The weights convey how much you can learn from a specific coalition. Large and Small coalitions mean more learned
because we've isolated more of the effect. At the same time, medium size coalitions don't supply us with as much
information because there are many possible such coalitions.

The main issue with the above is that there will be many possible coalitions, $2^M$ to be precise. Hence instead of
computing all of these, we use a sampling process on the space of coalitions and then estimate the Shapley values by
training a linear model. Because a coalition is a set of players/features that are contributing to a prediction, we
represent this as points in the space of binary codes $z' = \{z_0,...,z_m\}$ where $z_j = 1$ means that the $j^th$
feature is present in the coalition while $z_j = 0$ means it is not. To obtain the dataset on which we train this model,
we first sample from this space of coalitions then compute the values of $f$ for each sample. We obtain weights for each
sample using the Shapley Kernel:

$$ \pi_{x}(z') = \frac{M - 1}{\frac{M}{|z'|}|z'|(M - |z'|)} $$

Once we have the data points, the values of $f$ for each data point, and the sample weights, we have everything we need
to train a linear model. The paper shows that the coefficients of this linear model are the Shapley values.

There is some nuance to how we compute the value of a model given a specific coalition, as most models aren't built to
accept input with arbitrary missing values. If $D$ is the underlying distribution the samples are drawn from, then
ideally, we would use the conditional expectation:

$$ f(S) = \mathbb{E}_{D}[f(x)|x_S]
$$

Computing this value is very difficult. Instead, we can approximate the above using the interventional conditional
expectation, which is defined as:

$$ f(S) = \mathbb{E}_{D}[f(x)|do(x_S)]
$$

The $do$ operator here fixes the values of the features in $S$ and samples the remaining $\bar{S}$ feature values from
the data. A Downside of interfering in the distribution like this can mean introducing unrealistic samples if there are
dependencies between the features.

**Pros**

- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized
- Very general as is a blackbox method

**Cons**

- KernalSHAP is slow owing to the number of samples required to estimate the Shapley values accurately
- The interventional conditional probability introduces unrealistic data points
- Requires access to the training dataset

### TreeSHAP

| Model-types  | Task-types     | Data-types  |
| ------------ | -------------- | ----------- |
| Tree-based   | Classification | Tabular     |
|              | Regression     | Categorical |

In the case of tree-based models, we can obtain a speed-up by exploiting the structure of trees. Alibi exposes two
white-box methods, Interventional and Path dependent feature perturbation. The main difference is that the
path-dependent method approximates the interventional conditional expectation, whereas the interventional method
calculates it directly.

### Path Dependent TreeSHAP

Given a coalition, we want to approximate the interventional conditional expectation. We apply the tree to the features
present in the coalition like we usually would, with the only difference being when a feature is missing from the
coalition. In this case, we take both routes down the tree, weighting each by the proportion of samples from the
training dataset that go each way. For this algorithm to work, we need the tree to record how it splits the training
dataset. We don't need the dataset itself, however, unlike the interventional TreeSHAP algorithm.

Doing this for each possible set $S$ involves $O(TL2^M)$ time complexity. We can significantly improve the algorithm to
polynomial-time by computing the path of all sets simultaneously. The intuition here is to imagine standing at the first
node and counting the number of subsets that will go one way, the number that will go the other, and the number that
will go both (in the case of missing features). Because we assign different sized subsets different weights, we also
need to distinguish the above numbers passing into each tree branch by their size. Finally, we also need to keep track
of the proportion of sets of each size in each branch that contains a feature $i$ and the proportion that don't. Once
all these sets have flowed down to the leaves of the tree, then we can compute the Shapley values. Doing this gives us
$O(TLD^2)$ time complexity.

**Pros**

- Very fast for a valuable category of models
- Doesn't require access to the training data
- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized

**Cons**

- Only applies to Tree-based models
- Uses an approximation of the interventional conditional expectation instead of computing it directly

### Interventional Tree SHAP

The interventional TreeSHAP method takes a different approach. Suppose we sample a reference data point, $r$, from the
training dataset. Let $F$ be the set of all features. For each feature, $i$, we then enumerate over all subsets of
$S\subset F \setminus \{i\}$. If a subset is missing a feature, we replace it with the corresponding one in the
reference sample. We can then compute $f(S)$ directly for each coalition $S$ to get the Shapley values. One major
difference here is combining each $S$ and $r$ to generate a data point. Enforcing independence of the $S$ and
$F\setminus S$ in this way is known as intervening in the underlying data distribution and is where the algorithm's name
comes from. Note that this breaks any independence between features in the dataset, which means the data points we're
sampling won't be realistic.

For a single Tree and sample $r$ if we iterate over all the subsets of $S \subset F \setminus \{i\}$, the interventional
TreeSHAP method runs with $O(M2^M)$. Note that there are two paths through the tree of particular interest. The first is
the instance path for $x$, and the second is the sampled/reference path for $r$. Computing the Shapley value estimate
for the sampled $r$ will involve replacing $x$ with values of $r$ and generating a set of perturbed paths. Instead of
iterating over the sets, we sum over the paths. Doing so is faster as many of the routes within the tree have
overlapping components. We can compute them all at the same time instead of one by one. Doing this means the
Interventional TreeSHAP algorithm obtains $O(LD)$ time complexity.

Applied to a random forest with $T$ trees and using $R$ samples to compute the estimates, we obtain $O(TRLD)$ time
complexity. The fact that we can sum over each tree in the random forest results from the linearity property of Shapley
values.

**Pros**

- Very fast for a valuable category of models
- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized
- Computes the interventional conditional expectation exactly unlike the path-dependent method

**Cons**

- Less general as a white-box method
- Requires access to the dataset
- Typically, slower than the path-dependent method

### Counterfactuals

Given an instance of the dataset and a prediction given by a model, a question naturally arises how would the instance
minimally have to change for a different prediction to be provided. Counterfactuals are local explanations as they
relate to a single instance and model prediction.

Given a classification model trained on the MNIST dataset and a sample from the dataset with a given prediction, a
counterfactual would be a generated image that closely resembles the original but is changed enough that the model
correctly classifies it as a different number.

__TODO__:

- Give example image to illustrate

Similarly, given tabular data that a model uses to make financial decisions about a customer, a counterfactual would
explain how to change their behavior to obtain a different conclusion. Alternatively, it may tell the Machine Learning
Engineer that the model is drawing incorrect assumptions if the recommended changes involve features that are irrelevant
to the given decision.

A counterfactual, $x_{cf}$, needs to satisfy

- The model prediction on $x_{cf}$ needs to be close to the predefined output.
- The counterfactual $x_{cf}$ should be interpretable.

The first requirement is easy enough to satisfy. The second, however, requires some idea of what interpretable means.
Intuitively it would require that the counterfactual construction makes sense as an instance of the dataset. Each of the
methods available in Alibi deals with interpretability slightly differently. All of them require that the perturbation
$\delta$ that changes the original instance $x_0$ into $x_{cf} = x_0 + \delta$ should be sparse. Meaning, we prefer
solutions that change a small subset of the features to construct $x_{cf}$. Requiring this limits the complexity of the
solution making it more understandable.

:::{admonition} **Note 3: fit and explain method runtime differences**
Alibi explainers expose two methods, `fit` and `explain`. Typically, in machine learning, the method that takes the most
time is the fit method, as that's where the model optimization conventionally takes place. In explainability, the
explain step often requires the bulk of computation. However, this isn't always the case.

Among the explainers in this section, there are two approaches taken. The first fits a counterfactual when the user
requests the insight. This happens during the `.explain()` method call on the explainer class. This is done by running
gradient descent on model inputs to find a counterfactual. The methods that take this approach are counterfactual
instances, contrastive explanation, and counterfactuals guided by prototypes methods. Thus, the `fit` method in these
cases are quick, but the `explain` method is slow.

The other approach, however, uses reinforcement learning to train a model that produces explanations on demand. The
training takes place during the `fit` method call, so this has a long runtime while the `explain` method is quick. If
you want performant explanations in production environments, then the latter approach is preferable.
:::

__TODO__:

- schematic image explaining search for counterfactual as determined by loss
- schematic image explaining difference between different approaches.

### Counterfactuals Instances

| Model-types | Task-types     | Data-types  |
| ----------- | -------------- | ----------- |
| TF/Kera     | Classification | Tabular     |
| Black-box   |                | Image       |

Let the model be given by $f$, and let $p_t$ be the target probability of class $t$. Let $0<\lambda<1$ be a
hyperparameter. This method constructs counterfactual instances from an instance $X$ by running gradient descent on a
new instance $X'$ to minimize the following loss.

$$L(X', X)= (f_{t}(X') - p_{t})^2 + \lambda L_{1}(X', X)$$

The first term pushes the constructed counterfactual towards the desired class, and the use of the $L_{1}$ norm
encourages sparse solutions.

This method requires computing gradients of the loss in the model inputs. If we have access to the model and the
gradients are available, this can be done directly. If not, we can use numerical gradients, although this comes at a
considerable performance cost.

A problem arises here in that encouraging sparse solutions doesn't necessarily generate interpretable counterfactuals.
This happens because the loss doesn't prevent the counterfactual solution from moving off the data distribution. Thus,
you will likely get an answer that doesn't look like something that you'd expect to see from the data.

__TODO:__

- Picture example. Something similar to: https://github.com/interpretml/DiCE

**Pros**

- Both a black and white-box method
- Doesn't require access to the training dataset

**Cons**

- Not likely to give human interpretable instances
- Requires configuration in the choice of $\lambda$

### Contrastive Explanation Method

| Model-types | Task-types     | Data-types  |
| ----------- | -------------- | ----------- |
| TF/Kera     | Classification | Tabular     |
| Black-box   |                | Image       |

CEM follows a similar approach to the above but includes three new details. Firstly an elastic net $\beta L_{1} + L_{2}$
regularizer term is added to the loss. This term causes the solutions to be both close to the original instance and
sparse.

Secondly, we require that $\delta$ only adds new features rather than takes them away. We need to define what it means
for a feature to be present so that the perturbation only works to add and not remove them. In the case of the MNIST
dataset, an obvious choice of "present" feature is if the pixel is equal to 1 and absent if it is equal to 0. This is
simple in the case of the MNIST data set but more difficult in complex domains such as color images.

Thirdly, by training an optional autoencoder to penalize counter factual instances that deviate from the data
distribution. This works by minimizing the reconstruction loss of the autoencoder applied to instances. If a generated
instance is unlike anything in the dataset, the autoencoder will struggle to recreate it well, and its loss term will be
high. We require two hyperparameters $\beta$ and $\gamma$ to define the following Loss,

$$L(X'|X)= (f_{t}(X') - p_{t})^2 + \beta L_{1}(X', X) + L_{2}(X', X)^2 + \gamma L_{2} (X', AE(X'))^2$$

This approach extends the definition of interpretable to include a requirement that the computed counterfactual be
believably a member of the dataset. It turns out that minimizing this loss isn't enough to always get interpretable
results. And in particular, the constructed counterfactual often doesn't look like a member of the target class.

Similar to the previous method, this method can apply to both black and white-box models. In the black-box case, there
is still a performance cost from computing the numerical gradients.

__TODO:__

- Picture example of results including less interpretable ones.

**Pros**

- Provides more interpretable instances than the counterfactual instances' method.

**Cons**

- Requires access to the dataset to train the autoencoder
- Requires setup and configuration in choosing $\gamma$ and $\beta$ and training the autoencoder
- Requires domain knowledge when choosing what it means for a feature to be present or not

### Counterfactuals Guided by Prototypes

| Model-types | Task-types     | Data-types  |
| ----------- | -------------- | ----------- |
| TF/Kera     | Classification | Tabular     |
| Black-box   |                | Image       |
|             |                | Categorical |

- Black/white-box method
- Classification models
- Tabular, image and categorical data types

For this method, we add another term to the loss that optimizes for the distance between the counterfactual instance and
close members of the target class. In doing this, we require interpretability also to mean that the generated
counterfactual is believably a member target class and not just in the data distribution.

With hyperparameters $c$, $\gamma$ and $\beta$, the loss is now given by:

$$ L(X'|X)= c(f_{t}(X') - p_{t})^2 + \beta L_{1}(X', X) + L_{2}(X', X)^2 + \gamma L_{2} (X', AE(X'))^2 + L_{2}(X', X_
{proto})
$$

__TODO:__

- Picture example of results.

This method produces much more interpretable results. As well as this, because the proto term pushes the solution
towards the target class, we can remove the prediction loss term and still obtain a viable counterfactual. This doesn't
make much difference if we can compute the gradients directly from the model. If not, and we are using numerical
gradients, then the $L_{pred}$ term is a significant bottleneck owing to repeated calls on the model to approximate the
gradients. Thus, this method also applies to black-box models with a substantial performance gain on the previously
mentioned approaches.

**Pros**

- Generates more interpretable instances that the CEM method
- Blackbox version of the method doesn't require computing the numerical gradients
- Applies to more data-types

**Cons**

- Requires access to the dataset to train the auto encoder or k-d tree
- Requires setup and configuration in choosing $\gamma$, $\beta$ and $c$

### Counterfactuals with Reinforcement Learning

| Model-types | Task-types     | Data-types  |
| ----------- | -------------- | ----------- |
| Black-box   | Classification | Tabular     |
|             |                | Image       |
|             |                | Categorical |

This black-box method splits from the approach taken by the above three significantly. Instead of minimizing a loss
during the explain method call, it trains a new model when fitting the explainer called an actor that takes instances
and produces counterfactuals. It does this using reinforcement learning. In reinforcement learning, an actor model takes
some state as input and generates actions; in our case, the actor takes an instance with a target classification and
attempts to produce an member of the target class. Outcomes of actions are assigned rewards dependent on a reward
function designed to encourage specific behaviors. In our case, we reward correctly classified counterfactuals generated
by the actor. As well as this, we reward counterfactuals that are close to the data distribution as modeled by an
autoencoder. Finally, we require that they are sparse perturbations of the original instance. The reinforcement training
step pushes the actor to take high reward actions. CFRL is a black-box method as the process by which we update the
actor to maximize the reward only requires estimating the reward via sampling the counterfactuals.

As well as this, CFRL actors can be trained to ensure that certain constraints can be taken into account when generating
counterfactuals. This is highly desirable as a use case for counterfactuals is to suggest the necessary changes to an
instance to obtain a different classification. In some cases, you want these changes to be constrained, for instance,
when dealing with immutable characteristics. In other words, if you are using the counterfactual to advise changes in
behavior, you want to ensure the changes are enactable. Suggesting that someone needs to be two years younger to apply
for a loan isn't very helpful.

The training process requires randomly sampling data instances, along with constraints and target classifications. We
can then compute the reward and update the actor to maximize it. We do this without needing access to the model
internals; we only need to obtain a prediction in each case. The end product is a model that can generate interpretable
counterfactual instances at runtime with arbitrary constraints.

__TODO__:

- Example images

**Pros**

- Generates more interpretable instances that the CEM method
- Very fast at runtime
- Can be trained to account for arbitrary constraints
- General as is a black-box algorithm

**Cons**

- Longer to fit the model
- Requires to fit an autoencoder
- Requires access to the training dataset

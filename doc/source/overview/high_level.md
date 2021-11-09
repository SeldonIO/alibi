# Practical Overview of Explainability

While the applications of machine learning are impressive many models provide predictions that are hard to interpret or reason about. This limits their use in many cases as often we want to know why and not just what a model made a prediction. Alarming predictions are rarely taken at face value and typically warrant further analysis. Indeed, in some
cases explaining the choices that a model makes could even become a potential [legal requirement](https://arxiv.org/pdf/1711.00399.pdf). The following is a non-rigorous and practical overview of explainability and the methods alibi provide.

**Explainability provides us with algorithms that give insights into trained models predictions.** It allows
us to answer questions such as:
- How does a prediction change dependent on feature inputs?
- What features are Important for a given prediction to hold?
- What features are not important for a given prediction to hold?
- What set of features would you have to minimally change to obtain a new prediction of your choosing?
- How does each feature contribute to a model's prediction?

The set of insights available are dependent on the trained model. For instance, If the model is a regression it makes sense to ask how the prediction varies for some feature. Whereas, it doesn't make sense to ask what minimal change is required to obtain a new classification. Insights are constrained by:

- The type of data the model handles. Some insights only apply to image data others only to textual data
- The task the model performs. The two types of model tasks alibi handles are regression and classification
- The type of model used. Examples of model types include neural networks and random forests

Some explainer methods apply only to specific types of models such as TreeSHAP which can only be used with tree-based models. This is the case when an explainer method uses some aspect of that model's structure. If the model is a neural network then some methods require computing the prediction derivative for model inputs. Methods that require access to the model internals like this are known as **white box** methods. Other explainers apply to any type of model. They can do so because the underlying method doesn't make use of the model internals. Instead, they only need to have access to the model outputs given particular inputs. Methods that apply in this general setting are known as **black box** methods. Note that white box methods are a subset of black-box methods and an explainer being a white box method is a much stronger constraint than it being a black box method. Typically, white box methods are faster than black-box methods as access to the model internals means the method can exploit some aspect of that model's structure.


:::{admonition} **Note 1: Black Box Definition**
The use of Black Box here varies subtly from the conventional use within machine learning. Typically, we say a model is a black box if the mechanism by which it makes predictions is too complicated to be interpretable to a human. Here we use black-box to mean that the explainer method doesn't need access to the model internals to be applied.
:::

## Applications:

**Trust:** At a core level, explainability builds trust in the machine learning systems we use. It allows us to justify their use in many contexts where an understanding of the basis of the decision is paramount.

**Testing:** Explainability can be thought of as an extra form of testing for a model. The insights derived should conform to the expected behaviour. Failure to do so may indicate issues with the model or problems with the dataset it's been trained on.

**Functionality:** Insights can be used to augment model functionality. For instance, providing information on top of model predictions such as how to change model inputs to obtain desired outputs.

**Research:** Explainability allows researchers to understand how and why opaque models make decisions. Helping them understand more broadly the effects of the particular model or training schema they're using.

__TODO__:
- picture of explainability pipeline: training -> prediction -> insight

:::{admonition} **Note 2: Biases**

Practitioners must be wary of using explainability to excuse incorrect models rather than ensuring their correctness. It is possible to have a correctly trained model on a dataset, but the model doesn't reflect reality because the dataset is either wrong or incomplete. Suppose the insights that explainability generates conform to some confirmation bias of the person training the model. In that case, they will be blind to this issue and instead use these methods to confirm erroneous results. A key distinction is that explainability insights are designed to be faithful to the model they are explaining and not the data. You use an explanation to obtain an insight into the data only if the model is trained well.
:::

## Global and Local Insights

Insights can be categorised into two types. Local and global. Intuitively, a local insight says something about a single prediction that a model makes. For example, given an image classified as a cat by a model, a local insight might give the set of features (pixels) that need to stay the same for that image to remain classified as a cat. Such an insight provides an idea of what the model is looking for when classifying a specific instance into a particular class.

On the other hand, global insights refer to the behaviour of the model over a set of inputs. Plots showing how a regression prediction varies for a given feature while factoring out all the others are examples. These insights provide a more general understanding of the relationship between inputs and model predictions.

__TODO__:
- Add image to give idea of Global and local insight

## Insights

Alibi provides several local and global insights with which to explore and understand models. The following gives the practitioner an understanding of which explainers are suitable in which situations.

### Local Necessary Features:

Given a single instance and model prediction, local necessary features are local explanations that tell us what minimal set of features needs to stay the same so that the model still gives the same or close prediction.

In the case of a trained image classification model, local necessary features for a given instance would be a minimal subset of the image that the model uses to make its decision. A machine learning engineer might use this insight to see if the model concentrates on the correct features. Local necessary features are particularly advantageous for checking erroneous model decisions.

The following two explainer methods are available from alibi for generating Local Necessary Features insights. Each approaches the idea in slightly different ways. The main difference is that anchors give a picture of the size of the area over the dataset for which the insight applies, whereas pertinent positives do not.

#### Anchors

Anchors are a local blackbox method introduced in [Anchors: High-Precision Model-Agnostic Explanations](
https://homes.cs.washington.edu/~marcotcr/aaai18.pdf).

Let A be a rule (set of predicates) acting on such an interpretable representation, such that $A(x)$ returns $1$ if all its feature predicates are true. An example of such a rule, $A$, could be represented by the set $\{not, bad\}$ in which case any sentence, $s$, with both $not$ and $bad$ in it would mean $A(s)=1$.

Given a classifier $f$, a value $\tau>0$, an instance $x$, and a data distribution $\mathcal{D}$, $A$ is an anchor for $x$ if $A(x) = 1$ and,

$$
E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}] \geq \tau
$$

The distribution $\mathcal{D}(z|A)$ is those points from the dataset for which the anchor holds. This is like fixing some set of features of an instance and allowing all the others to vary. This condition says any point in the data distribution that satisfies the anchor $A$ is expected to match the model prediction $f(x)$ with probability $\tau$ (usually $\tau$ is chosen to be 0.95).

Let $prec(A) = E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}]$ be the precision of an anchor. Note that the precision of an anchor is considered with respect to the set of points in the data distribution to which the anchor applies, $\mathcal{D}(z|A)$. We can consider the **coverage** of an anchor as the probability that $A(z)=1$ for any instance $z$ in the data distribution. The coverage tells us the proportion of the distribution to which the anchor applies. The aim here is to find the anchor that applies to the most extensive set of instances while satisfying $prec(A) \geq \tau$.

__TODO__:
- Include picture explanation of anchors

Alibi finds anchors by building them up from the bottom up. We start with an empty anchor $A=\{\}$ and then consider the set of possible feature values from the instance of interest we can add. $\mathcal{A} = \{A \wedge a_0, A \wedge a_1, A \wedge a_2, ..., A \wedge a_n\}$. For instance, in the case of textual data, the $a_i$ might be words from the instance sentence. In the case of image data, we partition the image into super-pixels which we choose to be the $a_i$. At each stage, we will look at the set of possible next anchors that can be made by adding in a feature $a_i$ from the instance. We then compute the precision of each of the resulting anchors and choose the best. We iteratively build the anchor up like this until the required precision is met.

As we construct the new set of anchors from the last, we need to compute the precisions of the next group to know which one to choose. We can't calculate this directly; instead, we sample from $\mathcal{D}(z|A)$ to obtain an estimate. To do this, we use several different methods for different data types. For instance, in the case of textual data, we want to generate sentences with the anchor words and ensure that they make sense within the data distribution. One option is to replace the missing words (those not in the anchor) from the instance with words with the same POS tag. This means the resulting sample will make semantic sense rather than being a random set of words. Other methods are available. In the case of images, we sample an image from the data set and then superimpose the super-pixels on top of it.

**Pros**
- Easy to explain as rules are simple to interpret
- Is a black-box method as we just need to predict the value of an instance and don't need to access model internals
- Can be parallelized and made to be much more efficient as a result
- Although they apply to a local instance, the notion of coverage also gives a level of global insight

**Cons**
- This algorithm is much slower for high dimensional feature spaces
- If choosing an anchor close to a decision boundary, the method may require a considerable number of samples from $\mathcal{D}(z|A)$ and $\mathcal{D}(z|A')$ to distinguish two anchors $A$ and $A'$.
- High dimensional feature spaces such as images need to be reduced. Typically, this is done by segmenting the image into superpixels. The choice of the algorithm used to obtain the superpixels has an effect on the anchor obtained.
- Practitioners need to make several choices concerning parameters and domain-specific setup. For instance, the precision threshold or the method by which we sample $\mathcal{D}(z|A)$. Fortunately, alibi provides default settings for a lot of specific data types.


#### Pertinent Positives

Informally a Pertinent Positive is the subset of an instance that still obtains the same classification. These differ from anchors primarily in the fact that they aren't constructed to maximize coverage. The method to create them is also substantially different. The rough idea is to define an absence of a feature and then perturb the instance to take away as much information as possible while still retaining the original classification.

Given an instance $x_0$ we set out to find a $\delta$ that minimizes the following loss:

$$
L = c\cdot L_{pred}(\delta) + \beta L_{1}(\delta) + L_{2}^{2}(\delta) + \gamma \|\delta - AE(\delta)\|^{2}_{2}
$$

where

$$
L_{pred}(\delta) = max\left\{ max_{i\neq t_{0}}[Pred(\delta)]_{i} - [Pred(\delta)]_{t_0}, \kappa \right\}
$$

$AE$ is an optional autoencoder generated from the training data. If delta strays from the original data distribution, the autoencoder loss will increase as it will no longer reconstruct $\delta$ well. Thus, we ensure that $\delta$ remains close to the original dataset distribution.

Note that $\delta$ is restrained to only take away features from the instance $x_0$. There is a slightly subtle point here: removing features from an instance requires correctly defining non-informative feature values. In the case of MNIST digits, it's reasonable to assume that the black background behind each digit represents an absence of information. Similarly, in the case of color images, you might take the median pixel value to convey no information, and moving away from this value adds information.

Note that we need to compute the loss gradient through the model. If we have access to the internals, we can do this directly. Otherwise, we need to use numerical differentiation at a high computational cost due to the extra model calls we need to make.

**Pros**
- Both a black and white box method
- We obtain more interpretable results using the autoencoder loss as $\delta$ will be in the data distribution

**Cons**
- Finding non-informative feature values to add or take away from an instance is often not trivial, and domain knowledge is essential
- Need to tune hyperparameters $\beta$ and $\gamma$
- The insight doesn't tell us anything about the coverage of the pertinent positive
- If using the autoencoder loss, then we need access to the original dataset

### Local Feature Attribution

Local feature attribution asks how each feature in a given instance contributes to its prediction. In the case of an image, this would highlight those pixels that make the model provide the output it does. Note that this differs subtly from Local Necessary Features, which find the minimum subset of features required to give a prediction. Local feature attribution instead assigns a score to each feature.

__TODO__:
- picture showing above.

A good example use of local feature attribution is to detect that a classifier trained on images is focusing on the correct features of an image to infer the class. Suppose you have a model trained to classify breeds of dogs. You want to check that it focuses on the correct features of the dog in making its prediction. Suppose you compute the feature attribution of a picture of a husky and discover that the model is only focusing on the snowy backdrop to the husky, then you know two things. All the images of huskies in your dataset overwhelmingly have snowy backdrops, and also that the model will fail to generalize. It will potentially incorrectly classify other dog breeds with snowy backdrops as huskies and fail to recognize huskies that aren't in snowy locations.

Each of the following methods defines local feature attribution slightly differently. In both, however, we assign attribution values to each feature to indicate how significant those features were in making the model prediction.

Let $f:\mathbb{R}^n \rightarrow \mathbb{R}$. $f$ might be a regression, a single component of a multi regression or a probability of a class in a classification model. If $x=(x_1,... ,x_n) \in \mathbb{R}^n$ then an attribution of the prediction at input $x$ is a vector $a=(a_1,... ,a_n) \in \mathbb{R}^n$ where $a_i$ is the contribution of $x_i$ to the prediction $f(x)$.

The attribution values should satisfy specific properties:

1. Efficiency/Completeness: The sum of attributions equals the difference between the prediction and the baseline/average. We're interested in understanding the difference each feature value makes in a prediction compared to some uninformative baseline.
2. Symmetry: If the model behaves the same after swapping two variables $x$ and $y$, then $x$ and $y$ have equal attribution. If this weren't the case, we would be biasing the attribution towards certain features over other ones.
3. Dummy/Sensitivity: If a variable does not change the output of the model, then it should have attribution 0. If this were not the case, we'd be assigning value to a feature that provides no information.
4. Additivity/Linearity: The attribution for a feature $x_i$ of a linear composition of two models $f_1$ and $f_2$ given by $c_1 f_1 + c_2 f_2$ is $c_1 a_{1, i} + c_2 a_{2, i}$ where $a_{1, i}$ and $a_{2, i}$ is the attribution for $x_1$ and $f_1$ and $f_2$ respectively.

#### Integrated Gradients

This method computes the attribution of each feature by integrating the model partial derivatives along a path from a baseline point to the instance. Let $f$ be the model and $x$ the instance of interest. If $f:\mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ where $m$ is the number of classes the model predicts then let $F=f_k$ where $k \in \{1,..., m\}$. If $f$ is single-valued then $F=f$. We also need to choose a baseline value, $x'$.

$$
IG_i(x) = (x_i - x_i')\int_{\alpha}^{1}\frac{\partial F (x' + \alpha (x - x'))}{ \partial x_i } d \alpha
$$

The above sums partial derivatives for each feature over the path between the baseline and instance of interest. In doing so, you accumulate the changes in the prediction that occur due to the changing feature value from the baseline to the instance.

:::{admonition} **Note 5: Choice of Baseline**
The main difficulty with this method is that as IG is very dependent on the baseline, it's essential to make sure you choose it well. The choice of baseline should capture a blank state in which the model makes essentially no prediction or assigns the probability of each class equally. A common choice for image classification is an image set to black, which works well in many cases but sometimes fails to be a good choice. For instance, a model that classifies images taken at night using an image with every pixel set to black means the attribution method will undervalue the use of dark pixels in attributing the contribution of each feature to the classification. This is due to the contribution being calculated relative to the baseline, which is already dark.
:::

**Pros**
- Simple to understand and visualize, especially with image data
- Doesn't require access to the training data

**Cons**
- White box method. Requires the partial derivatives of the model outputs with respect to inputs
- Requires choosing the baseline which can have a significant effect on the outcome (See Note 5)


#### KernelSHAP

Kernel SHAP is a method of computing the Shapley values for a model around an instance $x_i$. Shapley values are a game-theoretic method of assigning payout to players depending on their contribution to an overall goal. In this case, the players are the features, and their payout is the model prediction. To compute these values, we have to consider the marginal contribution of each feature over all the possible coalitions of feature players.

Suppose we have a regression model $f$ that makes predictions based on four features $X = \{X_1, X_2, X_3, X_4\}$ as input. A coalition is a group of features, say, the first and third features. For this coalition, its value is given by:

$$
val({1,3}) = \int_{\mathbb{R}}\int_{\mathbb{R}} f(x_1, X_2, x_3, X_4)d\mathbb{P}_{X_{2}X_{4}} - \mathbb{E}_{X}(f(X))
$$

Given a coalition, $S$, that doesn't include $x_i$, then the marginal contribution of $x_i$ is given by $val(S \cup x_i) - val(S)$. Intuitively this is the difference that the feature $x_i$ would contribute if it was to join that coalition. We are interested in the marginal contribution of $x_i$ over all possible coalitions with and without $x_i$. A Shapley value for the $x_i^{th}$ feature is given by the weighted sum

$$
\psi_j = \sum_{S\subset \{1,...,p\} \setminus \{j\}} \frac{|S|!(p - |S| - 1)!}{p!}(val(S \cup x_i) - val(S))
$$

The weights convey how much you can learn from a specific coalition. Large and Small coalitions mean more learned because we've isolated more of the effect. At the same time, medium size coalitions don't supply us with as much information because there are many possible such coalitions.

The main issue with the above is that there will be many possible coalitions, $2^M$ to be precise. Hence instead of computing all of these, we use a sampling process on the space of coalitions and then estimate the Shapley values by training a linear model. Because a coalition is a set of players/features that are contributing to a prediction, we represent this as points in the space of binary codes $z' = \{z_0,...,z_m\}$ where $z_j = 1$ means that the $j^th$ feature is present in the coalition while $z_j = 0$ means it is not. To obtain the dataset on which we train this model, we first sample from this space of coalitions then compute the values of $f$ for each sample. We obtain weights for each sample using the Shapley Kernel:

$$
\pi_{x}(z') = \frac{M - 1}{\frac{M}{|z'|}|z'|(M - |z'|)}
$$

Once we have the data points, the values of $f$ for each data point, and the sample weights, we have everything we need to train a linear model. The paper shows that the coefficients of this linear model are the Shapley values.

There is some nuance to how we compute the value of a model given a specific coalition, as most models aren't built to accept input with arbitrary missing values. If $D$ is the underlying distribution the samples are drawn from, then ideally, we would use the conditional expectation:

$$
f(S) = \mathbb{E}_{D}[f(x)|x_S]
$$

Computing this value is very difficult. Instead, we can approximate the above using the interventional conditional expectation, which is defined as:

$$
f(S) = \mathbb{E}_{D}[f(x)|do(x_S)]
$$

The $do$ operator here fixes the values of the features in $S$ and samples the remaining $\bar{S}$ feature values from the data. A Downside of interfering in the distribution like this can mean introducing unrealistic samples if there are dependencies between the features.

**Pros**
- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized
- Very general as is a blackbox method

**Cons**
- KernalSHAP is slow owing to the number of samples required to estimate the Shapley values accurately
- The interventional conditional probability introduces unrealistic data points
- Requires access to the training dataset


#### TreeSHAP

In the case of tree-based models, we can obtain a speed-up by exploiting the structure of trees. Alibi exposes two white-box methods, Interventional and Path dependent feature perturbation. The main difference is that the path-dependent method approximates the interventional conditional expectation, whereas the interventional method calculates it directly.

#### Path Dependent TreeSHAP

Given a coalition, we want to approximate the interventional conditional expectation. We apply the tree to the features present in the coalition like we usually would, with the only difference being when a feature is missing from the coalition. In this case, we take both routes down the tree, weighting each by the proportion of samples from the training dataset that go each way. For this algorithm to work, we need the tree to record how it splits the training dataset. We don't need the dataset itself, however, unlike the interventional TreeSHAP algorithm.

Doing this for each possible set $S$ involves $O(TL2^M)$ time complexity. We can significantly improve the algorithm to polynomial-time by computing the path of all sets simultaneously. The intuition here is to imagine standing at the first node and counting the number of subsets that will go one way, the number that will go the other, and the number that will go both (in the case of missing features). Because we assign different sized subsets different weights, we also need to distinguish the above numbers passing into each tree branch by their size. Finally, we also need to keep track of the proportion of sets of each size in each branch that contains a feature $i$ and the proportion that don't. Once all these sets have flowed down to the leaves of the tree, then we can compute the Shapley values. Doing this gives us $O(TLD^2)$ time complexity.

**Pros**
- Very fast for a valuable category of models
- Doesn't require access to the training data
- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized

**Cons**
- Only applies to Tree-based models
- Uses an approximation of the interventional conditional expectation instead of computing it directly

#### Interventional Tree SHAP

The interventional TreeSHAP method takes a different approach. Suppose we sample a reference data point, $r$, from the training dataset. Let $F$ be the set of all features. For each feature, $i$, we then enumerate over all subsets of $S\subset F \setminus \{i\}$. If a subset is missing a feature, we replace it with the corresponding one in the reference sample. We can then compute $f(S)$ directly for each coalition $S$ to get the Shapley values. One major difference here is combining each $S$ and $r$ to generate a data point. Enforcing independence of the $S$ and $F\setminus S$ in this way is known as intervening in the underlying data distribution and is where the algorithm's name comes from. Note that this breaks any independence between features in the dataset, which means the data points we're sampling won't be realistic.

For a single Tree and sample $r$ if we iterate over all the subsets of $S \subset F \setminus \{i\}$, the interventional TreeSHAP method runs with $O(M2^M)$. Note that there are two paths through the tree of particular interest. The first is the instance path for $x$, and the second is the sampled/reference path for $r$. Computing the Shapley value estimate for the sampled $r$ will involve replacing $x$ with values of $r$ and generating a set of perturbed paths. Instead of iterating over the sets, we sum over the paths. Doing so is faster as many of the routes within the tree have overlapping components. We can compute them all at the same time instead of one by one. Doing this means the Interventional TreeSHAP algorithm obtains $O(LD)$ time complexity.

Applied to a random forest with $T$ trees and using $R$ samples to compute the estimates, we obtain $O(TRLD)$ time complexity. The fact that we can sum over each tree in the random forest results from the linearity property of Shapley values.

**Pros**
- Very fast for a valuable category of models
- The Shapley values are fairly distributed among the feature values
- Shapley values can be easily interpreted and visualized
- Computes the interventional conditional expectation exactly unlike the path-dependent method

**Cons**
- Less general as a white box method
- Requires access to the dataset
- Typically, slower than the path-dependent method

### Global Feature Attribution

Global Feature Attribution methods aim to show the dependency of model output on a subset of the input features. They are a global insight as it describes the behavior of the model over the entire input space. For instance, Accumulated Local Effects plots obtain graphs that directly visualize the relationship between feature and prediction.

Suppose a trained regression model that predicts the number of bikes rented on a given day depending on the temperature, humidity, and wind speed. A global feature attribution plot for the temperature feature might be a line graph plotted against the number of bikes rented. In the bikes rented case, one would anticipate an increase in rentals up until a specific temperature and then a decrease after it gets too hot.

#### Accumulated Local Effects

Alibi only provides accumulated local effects plots because of the available global feature attribution methods they give the most accurate insight. Alternatives include Partial Dependence Plots. ALE plots work by averaging the local changes in a prediction at every instance in the data distribution. They then accumulate these differences to obtain a plot of prediction over the selected feature dependencies.

Suppose we have a model $f$ and features $X=\{x_1,... x_n\}$. Given a subset of the features $X_S$, we denote $X_C=X \setminus X_S$. We want to obtain the ALE-plot for the features $X_S$, typically chosen to be at most a set of dimension two to be visualized easily. For simplicity assume we have $X=\{x_1, x_2\}$ and let $X_S=\{x_1\}$ so $X_C=\{x_2\}$. The ALE of $x_1$ is defined by:

$$
\hat{f}_{S, ALE}(x_1) =
\int_{min(x_1)}^{x_1}\mathbb{E}\left[
\frac{\partial f(X_1, X_2)}{\partial X_1} | X_1 = z_1
\right]dz_1 - c_1
$$

The term within the integral computes the expectation of the model derivative in $x_1$ over the random variable $X_2$ conditional on $X_1=z_1$. By taking the expectation for $X_2$, we factor out its dependency. So now we know how the prediction $f$ changes local to a point $X_1=z_1$ independent of $X_2$. Integrating these changes over $x_1$ from a minimum value to the value of interest, we obtain the global plot of how the model depends on $x_1$. ALE-plots get their names as they accumulate (integrate) the local effects (the expected partial derivatives). Note that here we have assumed $f$ is differentiable. In practice, however, we compute the various quantities above numerically, so this isn't a requirement.

__TODO__:
- Add picture explaining the above idea.

For more details on accumulated local effects including a discussion on PDP-plots and M-plots see
[Motivation-and-definition for ALE](../methods/ALE.ipynb#Motivation-and-definition)

:::{admonition} **Note 4: Categorical Variables and ALE**
Note that because ALE plots require computing differences between variables, they don't naturally extend to categorical data unless there is a sensible ordering on the data. As an example, consider the months of the year. To be clear, this is only an issue if the variable you are taking the ALE for is categorical.
:::

**Pros**:
- ALE-plots are easy to visualize and understand intuitively
- Very general as it is a black box algorithm
- Doesn't struggle with dependencies in the underlying features, unlike PDP plots
- ALE plots are fast

**Cons**:
- Harder to explain the underlying motivation behind the method than PDP plots or M plots.
- Requires access to the training dataset.
- Unlike PDP plots, ALE plots do not work with Categorical data

### Counter Factuals:

Given an instance of the dataset and a prediction given by a model, a question naturally arises how would the instance minimally have to change for a different prediction to be provided. Counterfactuals are local explanations as they relate to a single instance and model prediction.

Given a classification model trained on the MNIST dataset and a sample from the dataset with a given prediction, a counterfactual would be a generated image that closely resembles the original but is changed enough that the model correctly classifies it as a different number.

__TODO__:
- Give example image to illustrate

Similarly, given tabular data that a model uses to make financial decisions about a customer, a counterfactual would explain how to change their behavior to obtain a different conclusion. Alternatively, it may tell the Machine Learning Engineer that the model is drawing incorrect assumptions if the recommended changes involve features that are irrelevant to the given decision.

A counterfactual, $x_{cf}$, needs to satisfy

- The model prediction on $x_{cf}$ needs to be close to the predefined output.
- The counterfactual $x_{cf}$ should be interpretable.

The first requirement is easy enough to satisfy. The second, however, requires some idea of what interpretable means. Intuitively it would require that the counterfactual construction makes sense as an instance of the dataset. Each of the methods available in alibi deals with interpretability slightly differently. All of them require that the perturbation $\delta$ that changes the original instance $x_0$ into $x_{cf} = x_0 + \delta$ should be sparse. Meaning, we prefer solutions that change a small subset of the features to construct $x_{cf}$. Requiring this limits the complexity of the solution making it more understandable.

:::{admonition} **Note 3: fit and explain method runtime differences**
Alibi explainers expose two methods, `fit` and `explain`. Typically, in machine learning, the method that takes the most time is the fit method, as that's where the model optimization conventionally takes place. In explainability, the explain step often requires the bulk of computation. However, this isn't always the case.

Among the explainers in this section, there are two approaches taken. The first fits a counterfactual when the user requests the insight. This happens during the `.explain()` method call on the explainer class. This is done by running gradient descent on model inputs to find a counterfactual. The methods that take this approach are counterfactual instances, contrastive explanation, and counterfactuals guided by prototypes methods. Thus, the `fit` method in these cases are quick, but the `explain` method is slow.

The other approach, however, uses reinforcement learning to train a model that produces explanations on demand. The training takes place during the `fit` method call, so this has a long runtime while the `explain` method is quick. If you want performant explanations in production environments, then the latter approach is preferable.
:::

__TODO__:
- schematic image explaining search for counterfactual as determined by loss
- schematic image explaining difference between different approaches.

#### Counterfactuals Instances

- Black/white box method
- Classification models
- Tabular and image data types

Let the model be given by $f$, and let $p_t$ be the target probability of class $t$. Let $0<\lambda<1$ be a hyperparameter. This method constructs counterfactual instances from an instance $X$ by running gradient descent on a new instance $X'$ to minimize the following loss.

$$L(X', X)= (f_{t}(X') - p_{t})^2 + \lambda L_{1}(X', X)$$

The first term pushes the constructed counterfactual towards the desired class, and the use of the $L_{1}$ norm encourages sparse solutions.

This method requires computing gradients of the loss in the model inputs. If we have access to the model and the gradients are available, this can be done directly. If not, we can use numerical gradients, although this comes at a considerable performance cost.

A problem arises here in that encouraging sparse solutions doesn't necessarily generate interpretable counterfactuals. This happens because the loss doesn't prevent the counterfactual solution from moving off the data distribution. Thus, you will likely get an answer that doesn't look like something that you'd expect to see from the data.

__TODO:__
- Picture example. Something similar to: https://github.com/interpretml/DiCE

**Pros**
- Both a black and white box method
- Doesn't require access to the training dataset

**Cons**
- Not likely to give human interpretable instances
- Requires configuration in the choice of $\lambda$


#### Contrastive Explanation Method

- White box method
- Classification models
- Tabular and image data types

CEM follows a similar approach to the above but includes three new details. Firstly an elastic net $\beta L_{1} + L_{2}$ regularizer term is added to the loss. This term causes the solutions to be both close to the original instance and sparse.

Secondly, we require that $\delta$ only adds new features rather than takes them away. We need to define what it means for a feature to be present so that the perturbation only works to add and not remove them. In the case of the MNIST dataset, an obvious choice of "present" feature is if the pixel is equal to 1 and absent if it is equal to 0. This is simple in the case of the MNIST data set but more difficult in complex domains such as color images.

Thirdly, by training an optional autoencoder to penalize counter factual instances that deviate from the data distribution. This works by minimizing the reconstruction loss of the autoencoder applied to instances. If a generated instance is unlike anything in the dataset, the autoencoder will struggle to recreate it well, and its loss term will be high. We require two hyperparameters $\beta$ and $\gamma$ to define the following Loss,

$$L(X'|X)= (f_{t}(X') - p_{t})^2 + \beta L_{1}(X', X) + L_{2}(X', X)^2 + \gamma L_{2} (X', AE(X'))^2$$

This approach extends the definition of interpretable to include a requirement that the computed counterfactual be believably a member of the dataset. It turns out that minimizing this loss isn't enough to always get interpretable results. And in particular, the constructed counterfactual often doesn't look like a member of the target class.

Similar to the previous method, this method can apply to both black and white box models. In the black-box case, there is still a performance cost from computing the numerical gradients.

__TODO:__
- Picture example of results including less interpretable ones.

**Pros**
- Provides more interpretable instances than the counterfactual instances' method.

**Cons**
- Requires access to the dataset to train the autoencoder
- Requires setup and configuration in choosing $\gamma$ and $\beta$ and training the autoencoder
- Requires domain knowledge when choosing what it means for a feature to be present or not

#### Counterfactuals Guided by Prototypes

- Black/white box method
- Classification models
- Tabular, image and categorical data types

For this method, we add another term to the loss that optimizes for the distance between the counterfactual instance and close members of the target class. In doing this, we require interpretability also to mean that the generated counterfactual is believably a member target class and not just in the data distribution.

With hyperparameters $c$, $\gamma$ and $\beta$, the loss is now given by:

$$
L(X'|X)= c(f_{t}(X') - p_{t})^2 + \beta L_{1}(X', X) + L_{2}(X', X)^2 + \gamma L_{2} (X', AE(X'))^2 +
L_{2}(X', X_{proto})
$$

__TODO:__
- Picture example of results.

This method produces much more interpretable results. As well as this, because the proto term pushes the solution towards the target class, we can remove the prediction loss term and still obtain a viable counterfactual. This doesn't make much difference if we can compute the gradients directly from the model. If not, and we are using numerical gradients, then the $L_{pred}$ term is a significant bottleneck owing to repeated calls on the model to approximate the gradients. Thus, this method also applies to black-box models with a substantial performance gain on the previously mentioned approaches.

**Pros**
- Generates more interpretable instances that the CEM method
- Blackbox version of the method doesn't require computing the numerical gradients
- Applies to more data-types

**Cons**
- Requires access to the dataset to train the auto encoder or k-d tree
- Requires setup and configuration in choosing $\gamma$, $\beta$ and $c$

#### Counterfactuals with Reinforcement Learning

This black box method splits from the approach taken by the above three significantly. Instead of minimizing a loss during the explain method call, it trains a new model when fitting the explainer called an actor that takes instances and produces counterfactuals. It does this using reinforcement learning. In reinforcement learning, an actor model takes some state as input and generates actions; in our case, the actor takes an instance with a target classification and attempts to produce an member of the target class. Outcomes of actions are assigned rewards dependent on a reward function designed to encourage specific behaviors. In our case, we reward correctly classified counterfactuals generated by the actor. As well as this, we reward counterfactuals that are close to the data distribution as modeled by an autoencoder.  Finally, we require that they are sparse perturbations of the original instance. The reinforcement training step pushes the actor to take high reward actions. CFRL is a black-box method as the process by which we update the actor to maximize the reward only requires estimating the reward via sampling the counterfactuals.

As well as this, CFRL actors can be trained to ensure that certain constraints can be taken into account when generating counterfactuals. This is highly desirable as a use case for counterfactuals is to suggest the necessary changes to an instance to obtain a different classification. In some cases, you want these changes to be constrained, for instance, when dealing with immutable characteristics. In other words, if you are using the counterfactual to advise changes in behavior, you want to ensure the changes are enactable. Suggesting that someone needs to be two years younger to apply for a loan isn't very helpful.

The training process requires randomly sampling data instances, along with constraints and target classifications. We can then compute the reward and update the actor to maximize it. We do this without needing access to the model internals; we only need to obtain a prediction in each case. The end product is a model that can generate interpretable counterfactual instances at runtime with arbitrary constraints.

__TODO__:
- Example images

**Pros**
- Generates more interpretable instances that the CEM method
- Very fast at runtime
- Can be trained to account for arbitrary constraints
- General as is a Black box algorithm

**Cons**
- Longer to fit the model
- Requires to fit an autoencoder
- Requires access to the training dataset


##### Example:

__TODO__:
- Give example that applies to all the explainer methods.

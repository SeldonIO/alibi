# Practical Overview of Explainability

While the applications of machine learning are impressive many models provide predictions that are hard to interpret or
reason about. This limits there use in many cases as often we want to know why and not just what a model made a 
prediction. Alarming predictions are rarely taken at face value and typically warrant further analysis. Indeed, in some 
cases explaining the choices that a model makes could even become a potential 
[legal requirement](https://arxiv.org/pdf/1711.00399.pdf). The following is a non-rigorous and practical overview of 
explainability and the methods alibi provide.

**Explainability provides us with algorithms that give insights into trained models predictions.** It allows
us to answer questions such as:
- How does a prediction change dependent on feature inputs?
- What features are Important for a given prediction to hold?
- What features are not important for a given prediction to hold?
- What set of features would you have to minimally change to obtain a new prediction of your choosing?
- How does each feature contribute to a models prediction?

The set of insights available are dependent on the trained model. For instance, If the model is a regression it makes 
sense to ask how the prediction varies with respect to some feature. Whereas, it doesn't make sense to ask what minimal
change is required to obtain a new classification. Insights are constrained by:

- The type of data the model handles. Some insights only apply to image data others only to textual data.
- The task the model performs. The two types of model tasks alibi handles are regression and classification. 
- The type of model used. Examples of model types include neural networks and random forests.

Some explainer methods apply only to specific types of model such as TreeSHAP which can only be used with tree based 
models. This is the case when an explainer method makes use of some specific aspect of that models structure. For 
instance, if the model is a neural network then some methods require being able to compute the prediction derivative 
with respect to model inputs. Methods that require access to the model internals like this are known as **white box** 
methods. Other explainers apply to any type of model. They can do so because the underlying method doesn't make use of 
the model internals. Instead, they only require to have access to the model outputs given particular inputs. Methods 
that apply in this general setting are known as **black box** methods. Note that white box methods are a subset of 
black box methods and in general an explainer being a white box method is a much stronger constraint than it being a
black box methods. Typically, white box methods are faster than blackbox methods as access to the model internals means 
the method can exploit some aspect of that models structure. 

:::{admonition} **Note 1: Black Box Definition**
The use of Black Box here varies subtly from the conventional use within machine learning. Typically, we say a model is
a black box if the mechanism by which it makes predictions is too complicated to be interpretable to a human. Here we
use black box to mean that the explainer method doesn't need to have access to the model internals in order to be 
applied. 
:::

## Applications:

**Trust:** At a core level explainability builds trust in the machine learning systems we use. It allows us to justify 
there use in many contexts where an understanding of the basis of decision is paramount.

**Testing:** Explainability can be thought as an extra form of testing for a model. The insights derived should conform 
to the expected behaviour. Failure to do so may indicate issues with the model or problems with the dataset it's been 
trained on.

**Functionality:** Insights can also be used to augment model functionality. Providing useful information on top of 
model predictions. How to change the model inputs to obtain a better output for instance.

**Research:** Explainability allows researchers to understand how and why opaque models make decisions. Helping them 
understand more broadly the effects of the particular model or training schema they're using.

__TODO__:
- picture of explainability pipeline: training -> prediction -> insight

:::{admonition} **Note 2: Biases**
Practitioners must be wary of using explainability to excuse bad models rather than ensuring there correctness. As an 
example its possible to have a model that is correctly trained on a dataset, however due to the dataset being either 
wrong or incomplete the model doesn't actually reflect reality. If the insights that explainability generates 
conform to some confirmation bias of the person training the model then they are going to be blind to this issue and
instead use these methods to confirm erroneous results. A key distinction here is that explainability insights are 
designed to be faithful to the model they are explaining and not the data. You use an explanation to obtain an insight 
into the data only if the model is trained well.
:::

## Global and Local Insights

Insights can be categorized into two types. Local and global. Intuitively, a local insight says something about a 
single prediction that a model makes. As an example, given an image classified as a cat by a model, a local insight 
might give the set of features (pixels) that need to stay the same in order for that image to remain classified as a 
cat. Such an insight gives an idea of what the model is looking for when deciding to classify a specific instance into 
a specific class. Global insights on the other hand refer to the behaviour of the model over a set of inputs. Plots 
that show how a regression prediction varies with respect to a given feature while factoring out all the others are an 
example. These insights give a more general understanding of the relationship between inputs and model predictions. 

__TODO__:
- Add image to give idea of Global and local insight

## Insights

Alibi provides a number of local and global insights with which to explore and understand models. The following gives
the practitioner an understanding of which explainers are suitable in which situations.

### Local Necessary Features:

Given a single instance and model prediction, Local Necessary Features are local explanations that tell us what minimal 
set of features needs to stay the same in order that the model still give the same or close prediction. This tells us 
what it is in an instance that most influences the result.

In the case of a trained image classification model, Local Necessary Features for a given instance would be a minimal 
subset of the image that the model uses to make its decision. A Machine learning engineer might use this insight to see 
if the model is concentrating on the correct features of an image in making a decision. This is especially useful 
applied to an erroneous decision.

The following two explainer methods are available from alibi for generating Local Necessary Features insights. Each 
approaches the idea in slightly different ways but the main difference is that anchors gives an idea of the size of the
area over the dataset for which the insight applies whereas pertinent positives do not.

#### Anchors

Anchors are a local blackbox method. This section takes the definition and discussion from [Anchors: High-Precision 
Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf).

Let A be a rule (set of predicates) acting on such an interpretable representation, such that $A(x)$ returns $1$ if all 
its feature predicates are true for instance $x$. An example of such a rule, $A$, could be represented by the set 
$\{not, bad\}$ in which case any sentence, $s$, with both $not$ and $bad$ in it would mean $A(s)=1$.

Given a classifier $f$, a value $\tau>0$, an instance $x$ and a data distribution $\mathcal{D}$, $A$ is an anchor for 
$x$ if $A(x) = 1$ and,

$$ E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}] ≥ \tau $$

The distribution $\mathcal{D}(z|A)$ is those points from the dataset for which the anchor holds. This is like fixing 
some set of features of an instance and allowing all the others to vary. This condition says any point in the data 
distribution that satisfies the anchor $A$ is expected to match the model prediction $f(x)$ with probability $\tau$ 
(usually $\tau$ is chosen to be 0.95). 

Let $prec(A) = E_{\mathcal{D}(z|A)}[1_{f(x)=f(z)}]$ be the precision of an anchor. Note that the precision of an anchor 
is considered with respect to the set of points in the data distribution to which the anchor applies,
$\mathcal{D}(z|A)$. We can consider the **coverage** of an anchor as the probability that $A(z)=1$ for any instance 
$z$ in the data distribution. The coverage tells us the proportion of the distribution that the anchor applies to. 
The aim here is to find the anchor that applies to the largest set of instances while still satisfying 
$prec(A) \geq \tau$.

__TODO__: 
- Include picture explanation of anchors

Alibi finds anchors by building them up from the bottom up. We start with an empty anchor $A=\{\}$ and then consider
the set of possible feature values from the instance of interest we can add. $\mathcal{A} = 
\{A \wedge a_0, A \wedge a_1, A \wedge a_2, ..., A \wedge a_n\}$. In the case of textual data for instance the $a_i$ 
might be words from the instance sentence. In the case of image data we take a partition the image into superpixels 
which we choose to be the $a_i$. At each stage we're going to look at the set of possible next anchors that can be made
by adding in a feature $a_i$ from the instance. We then compute the precision of each of the resulting anchors and 
choose the best. We iteratively build the anchor up like this until the required precision is met.
 
As we build up the anchors from the empty case we need to compute their precisions in order to know which one to choose. 
We can't compute this directly, instead we sample from $\mathcal{D}(z|A)$ to obtain an estimate. To do this we use a 
number of different methods for different data types. For instance, in the case of textual data we want to generate 
sentences with the anchor words in them and also ensure that they make sense within the data distribution. One option 
is to replace the missing words (those not in the anchor) from the instance with words with the same POS tag. This means
the resulting samples makes semantic sense rather than being a random set of words. Other methods are available. In the 
case of images we sample an image from the data set and then superimpose the superpixels on top of it.

**Pros**
- Easy to explain as rules are simple to interpret.
- Is a blackbox method as we just need to be able to predict the value of an instance and don't need to access model 
internals.
- Can be parallelized and made to be much more efficient as a result.
- Although they apply to a local instance, the notion of coverage also gives a level of global insight as well. 

**Cons**
- Because at each step in building up the anchor we need to consider all the possible features we can add this 
algorithm is slower for high dimensional feature spaces.
- If choosing an anchor close to a decision boundary the method may require a very large number of samples from 
$\mathcal{D}(z|A)$ and $\mathcal{D}(z|A')$ in order to distinguish two anchors $A$ and $A'$.
- High dimensional feature spaces such as images need to be reduced. Typically, this is done by segmenting the image 
into superpixels but the choice of algorithm used to obtain the superpixels has an effect on the anchor obtained.
- Practitioners need to make a number of choices with respect to parameters and domain specific setup. For instance the 
precision threshold or the method by which we sample $\mathcal{D}(z|A)$. Fortunately alibi provides default settings 
for a lot of specific data-types.

#### Pertinent Positives

Informally a Pertinent Positive is the subset of an instance that still obtains the same classification. These differ
from anchors primarily in the fact that they aren't constructed to maximize coverage. The method to obtain them is 
also substantially different.

Given an instance $x_0$ we set out to find a $\delta$ that minimizes the following loss:

$$
l = \left \{ 
c\cdot L_{pred}(\delta) + \beta L_{1}(\delta) + L_{2}^{2}(\delta) + \gamma \|\delta - AE(\delta)\|^{2}_{2} 
\right \}
$$

where

$$
L_{pred}(\delta) = max\left\{ max_{i\neq t_{0}}[Pred(\delta)]_{i} - [Pred(\delta)]_{t_0}, \kappa \right\}
$$

and $\delta$ is restrained to only take away features from the instance $x_0$. There is a slightly subtle point here in 
that removing features from an instance requires correctly defining non-informative feature values. In the case of
MNIST digits it's reasonable to assume that the black background behind each digit represents an absence of information.
Similarly, in the case of color images you might assume that the median pixel value represents no information and moving
away from this value adds information. It is however often not trivial to find these non-informative feature values and 
domain knowledge becomes very important.

Note this is both a black and white box method as while we need to compute the loss gradient through the model we can 
do this using numerical differentiation. This comes at a significant computational cost due to the extra model calls 
we need to make.

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
$X_C=X \setminus X_S$. We want to obtain the ALE-plot for the features $X_S$, typically chosen to be at most a
set of dimension 2 in order that they can easily be visualized. For simplicity assume we have $X=\{x_1, x_2\}$ and let 
$X_S=\{x_2\}$ so $X_C=\{x_1\}$. The ALE of $x_1$ is defined by:

$$
\hat{f}_{S, ALE}(x_1) = 
\int_{min(x_1)}^{x_1}\mathbb{E}\left[ 
\frac{\partial f(X_1, X_2)}{\partial X_1} | X_1 = z_1 
\right]dz_1 - c_1
$$

The term within the integral computes the expectation of the model derivative in $x_1$ over the random variable $X_2$ 
conditional on $X_1=z_1$. By taking the expectation with respect to $X_2$ we factor out its dependency. So now we know 
how the prediction $f$ changes local to a point $X_1=z_1$ independent of $X_2$. If we have this then to get the true 
dependency we must integrate these changes over $x_1$ from a min value to the value of interest. ALE-plots get there 
names as they accumulate (integrate) the local effects (the expected partial derivatives). 

__TODO__: 
- Add picture explaining the above idea.

It is important to note that by considering effects local to $X_1=z_1$ in the above equations we capture any 
dependencies in the features of the dataset. Similarly, by considering the differences in $f$ and accumulating them 
over the variable of interest we remove any effects owing to correlation between $X_1$ and $X_2$. For better insight 
into these points see... 

In the above example we've assumed $f$ is differentiable. In practice however, we compute the various quantities above 
numerically and so this isn't a requirement.

:::{admonition} **Note 4: Categorical Variables and ALE**
Note that because ALE plots require differences between variable they don't natural extend to categorical data unless
there is a sensible ordering on the categorical data. As an example consider months of the year. To be clear this is
only an issue if the variable you are taking the ALE with respect to is categorical. 
:::

#### Local Feature Attribution

Local feature attribution asks how each feature in a given instance contributes to its prediction. In the case of an 
image this would highlight those pixels that make the model give the output it does. Note this differs subtly from 
Local Necessary Features in that they find the minimum subset of features required to give a prediction whereas local 
feature attribution creates a heat map that tells us each features contribution to the overall outcome.

__TODO__:
- picture showing above concept.

A good use of local feature attribution is to detect that a classifier trained on images is focusing on the correct 
features of an image in order to infer the class. As an example suppose you have a model trained to detect breeds of 
dog. You want to check that it focuses on the correct features of the dog in making its prediction. If you compute
the feature attribution of a picture of a husky and discover that the model is only focusing on the snowy backdrop to
the husky then you know both that all the images of huskies in your dataset overwhelmingly have snowy backdrops and 
also that the model will fail to generalize. It will potentially incorrectly classify other breeds of dog with snowy 
backdrops as huskies and also fail to recognise huskies without snowy backdrops.

Each of the following methods defines local feature attribution slightly differently. In both however we assign 
attribution values to each feature to indicate how significant those features where in making the model prediction what 
it is. 

Let $f:\mathbb{R}^n \rightarrow \mathbb{R}$. $f$ might be a regression, a single component of a multi regression or a 
probability of a class in a classification model. If $x=(x_1,... ,x_n) \in \mathbb{R}^n$ then an attribution of the 
prediction at input $x$ is a vector $a=(a_1,... ,a_n) \in \mathbb{R}^n$ where $a_i$ is the contribution of $x_i$ to the 
prediction $f(x)$.

Its desirable that these attribution values satisfy certain properties:

1. Efficiency/Completeness: The sum of attributions equals the difference between the prediction and the
baseline/average. It makes sense that this be the case as we're interested in understanding the difference each feature
value makes in a prediction compared to some uninformative baseline.
2. Symmetry: If the model behaves the same after swapping two variables $x$ and $y$ then $x$ and $y$ have equal 
attribution. If this weren't the case we'd be biasing the attribution towards certain features over other ones.
3. Dummy/Sensitivity: If a variable does not change the output of the model then it should have attribution 0. Similar 
to above if this where not the case then we'd be assigning value to a feature that provides no information.
4. Additivity/Linearity: The attribution for a feature $x_i$ of a linear composition of two models $f_1$ and $f_2$ 
given by $c_1 f_1 + c_2 f_2$ is $c_1 a_{1, i} + c_2 a_{2, i}$ where $a_{1, i}$ and $a_{2, i}$ is the attribution for 
$x_1$ and $f_1$ and $f_2$ respectively. This allows us to decompose certain types of model that are made up of lots of
smaller models such as random forests.

Each of the methods that alibi provides satisfies these properties.

##### Explainers:

**Integrated Gradients**

The integrated gradients' method computes the attribution of each feature by integrating the model partial derivatives 
along a path from a baseline point to the instance. Let $f$ be the model and $x$ the instance of interest. 
If $f:\mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ where $m$ is the number of classes the model predicts then let $F=f_k$
where $k \in \{1,..., m\}$. If $f$ is single valued then $F=f$. We also need to choose a baseline value, $x'$.

$$
IG_i(x) = (x_i - x_i')\int_{\alpha}^{1}\frac{\partial F (x' + \alpha (x - x'))}{ \partial x_i } d \alpha
$$

So the above sums the partial derivatives with respect to each feature over the path between the baseline and the 
instance of interest. In doing so you accumulate the changes in prediction that occur as a result of the changing 
feature value from the baseline to the instance. 

:::{admonition} **Note 5: Choice of Baseline**
The main difficulty with this method ends up being that as IG is very dependent on the baseline it's important to make 
sure you choose the correct baseline. The choice of baseline should capture a blank state in which the model makes 
essentially no prediction or assigns probability of classes equally. A common choice for image classification is an 
image set to black. This works well in many cases but sometimes fails to be a good choice. For instance for a model 
that classifies images taken at night using an image with every pixel set to black means the attribution method will 
undervalue the use of dark pixels in attributing the contribution of each feature to the classification. This is due to 
the contribution being calculated relative to the baseline which in this case is already dark.
:::


**KernelSHAP**

Kernel SHAP is an efficient method of computing the Shapley values for a model around an instance $x_i$. Shapley values
are a game theoretic method of assigning payout to players depending on there contribution to an overall goal. In this
case the players are the features and the payout is the prediction the model makes. To compute these values we have to
consider the marginal contribution of each feature over all the possible coalitions of feature players. Exactly what 
this means in terms of a specific coalition is a little nuanced. Suppose for example we have a regression model $f$ 
that makes predictions based on four features $X = \{X_1, X_2, X_3, X_4\}$ as input. A coalition is a group of 
features, say for example the first and third feature. For this coalition it's value is given by:

$$
val({1,3}) = \int_{\mathbb{R}}\int_{\mathbb{R}} f(x_1, X_2, x_3, X_4)d\mathbb{P}_{X_{2}X_{4}} - \mathbb{E}_{X}(f(X))
$$

Given a coalition, $S$, that doesn't include $x_i$, then that features marginal contribution is given by 
$val(S \cup x_i) - val(S)$. Intuitively this is the difference that feature $x_i$ would make for that coalition. We are 
interested in the marginal contribution of $x_i$ over all possible coalitions with and without $x_i$. A shapley value 
for the $x_i^{th}$ feature is given by the weighted sum

$$
\psi_j = \sum_{S\subset \{1,...,p\} \setminus \{j\}} \frac{|S|!(p - |S| - 1)!}{p!}(val(S \cup x_i) - val(S))
$$

The motivation for the weights convey how much you can learn from a specific coalition. Large and Small coalitions mean 
more learnt because we've isolated more of the effect. Whereas medium size coalitions don't supply us with as much 
information because there are many possible such coalitions. **Not 100 percent sure about this point! Want a nicer 
intuitive explanation.**.

The main issue with the above is that there will be a large number of possible coalitions, $O(M2^M)$ to be precise. 
Hence instead of computing all of these we use a sampling process on the space of coalitions and then estimate the
Shapley values by training a linear model. Because a coalition is a set of players/features that are or aren't 
playing/contributing we represent this as points in the space of binary codes $z' = \{z_0,...,z_m\}$ where $z_j = 0$ 
means that feature is present or not. To obtain the dataset on which we train this model we first sample from this 
space of coalitions then compute the values of $f$ for each sample. We obtain weights for each sample using the Shapley
Kernel:

$$
\pi_{x}(z') = \frac{M - 1}{\frac{M}{|z'|}|z'|(M - |z'|)}
$$

Once we have all of this we can train a linear model, the coefficients of which are the Shapley values. There is some 
nuance to how we compute the value of a model given a specific coalition as most models aren't built to accept input 
with arbitrary missing values. Given a coalition $S$ we can either fix those features in $S$ and average out those not 
in $S$ by replacing them by values sampled from the data set. This is the marginal expectation given by 
$\mathbb{E}_{X_{X \setminus S}}[f(x_S, X_{X \setminus S})]$. A problem with this approach is that fixing the $x_S$ and
replacing the other inputs with feature values sampled from $X_{X \setminus S}$ can introduce unrealistic data if the
underlying data has dependencies. Another approach in this case is to use the conditional expectation given by 
$\mathbb{E}[f(x_S, X_{X\setminus S}|X_S=x_S)]$. This is still problematic however, as a variables that don't contribute 
to a prediction can be highly correlated with features that do. In this case the shapley value for the non-contributing 
feature can end up being assigned a none zero shapley value which violates the Dummy/Sensitivity property.

**TreeSHAP**

In the case of tree based models we can obtain a speed-up by exploiting the structure of trees. Alibi exposes two 
white-box methods Interventional and Path dependent feature perturbation. In each case the speed-up is in how we 
compute the value of a coalition.

Path Dependent: 

Given a coalition $S=/{z_1, ..., z_n/}$ we want to find $\mathbb{E}_{X_{X \setminus S}}[f(x_S, X_{X \setminus S})]$. We 
do so in the same way we should when applying a tree based model to any sample, with the only difference being what to
do when a feature is missing from the coalition. In this case we take both routes down the tree and weight each by the
proportion of dataset samples from the training dataset that go into each branch. For this algorithm to work we need 
the tree to have a record of how it splits the training dataset. We don't need the dataset itself however, unlike the 
interventional TreeSHAP algorithm.

Doing this for each possible set $S$ involves $O(TL2^M)$ time complexity. It can be significantly improved to polynomial 
time however if instead of doing the above one by one we do it for all subsets at once. The intuition here is to imagine
standing at the first node and counting the number of subsets that will go one way, the number that will go the other
and the number that will go both. Because we assign different sized subsets different weights we also need to further
distinguish the above numbers passing into each branch of the tree by there size. Finally, we also need to keep track of
the proportion of sets of each size in each branch that contain a feature $i$ and the proportion that don't. Once all 
these sets have flowed down to the leaves of the tree then we can compute the shapley values. Doing this gives us 
$O(TLD^2)$ time complexity.

An issue is that as we're approximating $\mathbb{E}[f(x_S, X_{X\setminus S}|X_S=x_S)]$ this method will give shapley 
values that don't satisfy the Dummy/Sensitivity property. This occurs because features that contribute nothing can be 
highly correlated with features that do and the expectation doesn't distinguish on this effect. Hence, the 
non-significant but correlated feature will have a positive shapley coefficient.

Interventional:

The interventional TreeSHAP method takes a different approach. Suppose we sample a reference data point, $r$, from the 
training dataset. For each feature $i$ we then enumerate over all subsets of $S\subset F \setminus \{i\}$. If a subset 
is missing a feature we replace it with the corresponding one in the reference sample. We can then compute $f(S)$ 
directly for each set $S$ and from that get the Shapley values. One major difference here is that we're combining each 
$S$ and $r$ to generate a data point. The process of enforcing independence of the $S$ and $F\S$ in this way is known
as intervening in the underlying data set and is where the algorithm name comes from. Note that doing this breaks 
any independence between features in the dataset which means the data points we're sampling won't be realistic.

For a single Tree and sample $r$ if we just iterate over all the subsets of $S \subset F \setminus \{i\}$ the 
interventional TreeSHAP method runs with $O(M2^M)$. Note that there are two paths through the tree of unique interest. 
The first is the instance path for $x$, and the second is the sampled/reference path for $r$. Computing the shapley 
value estimate for the sampled $r$ is going to involve replacing $x$ with values of $r$ and generating a set of 
perturbed paths. If, instead of summing over the sets if we sum over the paths we get a speed improvement as many of
the sets $S$ end up taking the same path, and we compute them all at the same time instead of one by one. Doing so the
Interventional TreeSHAP algorithm obtains $O(LD)$ time complexity.

Applied to a random forrest with $T$ trees and using $R$ samples to compute the estimates we obtain $O(TRLD)$ time 
complexity. The fact that we can sum over each tree in the random forest is a results of the linearity property of
shapley values.

__WARNING__: 
- Reference the Christoph Molnar book!

#### Counter Factuals:

Given an instance of the dataset and a prediction given by a model a question that naturally arises is how would the
instance minimally have to change in order for a different prediction to be given. Counterfactuals are local 
explanations as they relate to a single instance and model prediction.

Given a classification model trained on MNIST and a sample from the dataset with a given prediction, a counter factual 
would be a generated image that closely resembles the original but is changed enough that the model correctly 
classifies it as a different number.

Similarly, given tabular data that a model uses to make financial decisions about a customer a counter factual would
explain to a user how to change they're behaviour in order to obtain a different decision. Alternatively it may tell
the Machine Learning Engineer that the model is drawing incorrect assumptions if the recommended changes involve
features that shouldn't be relevant to the given decision. This may be down either to the model training or the dataset
being unbalanced.

A counterfactual, $x_{cf}$, needs to satisfy

- The model prediction on $x_{cf}$ needs to be close to the predefined output.
- The counterfactual $x_{cf}$ should be interpretable. 

The first requirement is easy enough to satisfy. The second however requires some idea of what interpretable means. 
Intuitively it would require that the counterfactual constructed makes sense as an instance of the dataset. Each of the
methods available in alibi deal with interpretability slightly differently. All of them agree however that we require 
that the perturbation $\delta$ changing the original instance $x_0$ into $x_{cf} = x_0 + \delta$ should be sparse. This 
means we prefer solutions that change a small subset of the features to construct $x_{cf}$. This limits the complexity
of the solution making them more understandable. 

##### Explainers:

The following discusses the set of explainer methods available from alibi for generating counterfactual insights. 

:::{admonition} **Note 3: fit and explain method runtime differences**
Alibi explainers expose two methods `fit` and `explain`. Typically, in machine learning the method that takes the most 
time is the fit method as that's where the model optimization conventionally takes place. In explainability however the 
model should already be fit and instead the explain step usually requires the bulk of computation. However this isn't
always the case.

Among the following explainers there are two categories of approach taken. The first fits a counterfactual when the 
user requests the insight. This happens during the `.explain()` method call on the explainer class. They do this by 
running gradient descent on model inputs to find a counterfactual. The methods that take this approach are 
Counterfactuals Instances, Contrastive Explanation Method and Counterfactuals Guided by Prototypes. Thus, the `fit`
methods in these cases are quick but the `explain` method slow.

The other approach however uses reinforcement learning to pretrains a model that produces explanations on the fly. The 
training in this case takes place during the `fit` method call and so this has a long runtime while the `explain` 
method is quick. If you want performant explanations in production environments then the later method is preferable.
:::

__TODO__: 
- schematic image explaining search for counterfactual as determined by loss
- schematic image explaining difference between different approaches.

**Counterfactuals Instances:**

- Black/white box method
- Classification models
- Tabular and image data types

Let the model be given by $f$ and $f_{t}$ be the probability of class $t$, $p_t$ is the target probability of class 
$t$ and $0<\lambda<1$ a hyperparameter. This method constructs counterfactual instances from an instance $X$ by running 
gradient descent on a new instance $X'$ to minimize the following loss.

$$L(X'|X)= (f_{t}(X') - p_{t})^2 + \lambda L_{1}(X'|X)$$ 

The first term pushes the constructed counterfactual towards the desired class and the use of the $L_{1}$ norm 
encourages sparse solutions. 

This method requires computing gradients of the loss in the model inputs. If we have access to the model and the 
gradients are available then this can be done directly. If not however, we can use numerical gradients although this
comes at a considerable performance cost.

A problem arises here in that encouraging sparse solutions doesn't necessarily generate interpretable counterfactuals. 
This happens because the loss doesn't prevent the counter factual solution moving off the data distribution. Thus, you
will likely get a solution that doesn't look like something that you'd expect to see from the data.

__TODO:__
- Picture example. Like from here: https://github.com/interpretml/DiCE

**Contrastive Explanation Method:**

- c/white box method
- Classification models
- Tabular and image data types

CEM follows a similar approach to the above but includes two new details. Firstly an elastic net 
($\beta L_{1} + L_{2}$) regularizer term is added to the loss. This causes the solutions to be both close to the 
original instance and sparse. Secondly an optional autoencoder is trained to penalize conterfactual instances that 
deviate from the data distribution. This works by requiring minimize the gradient descent minimize the reconstruction 
loss of instances passed through the autoencoder. If an instance is unlike anything in the dataset then the autoencoder 
will struggle to recreate it well, and it's loss term will be high. We require two hyperparameters $\beta$ and $\gamma$
to define the following Loss,

$$L(X'|X)= (f_{t}(X') - p_{t})^2 + \beta L_{1}(X' - X) + L_{2}(X' - X)^2 + \gamma L_{2} (X' - AE(X'))^2$$ 

This approach extends the definition of interpretable to include a requirement that the computed counterfactual be 
believably a member of the dataset. It turns out however that this condition isn't enough to always get interpretable 
results. And in particular the constructed counterfactual often doesn't look like a member of the target class. 

Similarly to the previous method, this method can apply to both black and white box models. In the case of black box 
there is still a performance cost from computing the numerical gradients.

__TODO:__
- Picture example of results including less interpretable ones.

**Counterfactuals Guided by Prototypes:**

- Black/white box method
- Classification models
- Tabular, image and categorical data types
 
For this method we add another term to the loss that optimizes for distance between the counterfactual instance and
close members of the target class. Now the definition of interpretability has been extended even further to include the
requirement that the counterfactual be believably a member of the target class and not just in the data distribution.

With hyperparmaters $c$ and $\beta$, the loss is now given by:

$$L(X'|X)= cL_{pred} + \beta L_{1} + L_{2} + L_{AE} + L_{proto}$$

__TODO:__ 
- Picture example of results.

It's clear that this method produces much more interpretable results. As well as this, because the proto term pushes
the solution towards the target class we can actually remove the prediction loss term and still obtain a viable counter
factual. This doesn't make much difference if we can compute the gradients directly from the model but if not, and we 
are using numerical gradients then the $L_{pred}$ term is a significant bottleneck owing to repeated calls on the model
to approximate the gradients. Thus, this method also applies to blackbox models with a significant performance gain on
the previous approaches mentioned.

**Counterfactuals with Reinforcement Learning:**

This black box method splits from the approach taken by the above three significantly. Instead of minimizing a loss at 
explain time it trains a new model when fitting the explainer called an actor that takes instances and produces 
counterfactuals. It does this using reinforcement learning. In RL an actor model takes some state as input and 
generates actions, in our case the actor takes an instance with a specific classification and produces an action in the 
form of a counter factual. Outcomes of actions are assigned reward dependent on some reward function that's designed to 
encourage specific behaviours of the actor. In our case we reward counterfactuals that are firstly classified as the 
correct target class, secondly are close to the data distribution as modelled by an auto-encoder, and thirdly are sparse 
perturbations of the original instance. The reinforcement training step pushes the actor to take high reward actions 
instead of low reward actions. This method is black box as we compute the reward obtained by an actor by sampling an 
action by directly predicting the loss. Because of this we only require the derivative with respect to the actor and 
not the predictor itself.

As well as this CFRL actors are trained to ensure that certain constraints can be taken into account when generating 
counterfactuals. This is highly desirable as a use case for counterfactuals is suggesting small changes to an 
instance in order to obtain a different classification. In certain cases you want these changes to be constrained, for
instance when dealing with immutable characteristics. In other words if you are using the counterfactual to advise 
changes in behaviour you want to ensure the changes are enactable. Suggesting that someone needs to be 2 years younger
to apply for a loan isn't very helpful.

To train the actor we randomly generate in distribution data instances, along with constraints and target 
classifications. We then compute the reward and update the actor to maximize the reward. We do this without needing 
access to the critic internals. We end up with an actor that is able to generate interpretable counterfactual instances 
at runtime with arbitrary constraints. 

__TODO__: 
- Example images
___



##### Example:

__TODO__: 
- Give example that applies to all the explainer methods. 
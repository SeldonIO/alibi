# Anchor explanations for movie sentiment

In this example, we will explain why a certain sentence is classified by a logistic regression as having negative or positive sentiment. The logistic regression is trained on negative and positive movie reviews.

Note

To enable support for the anchor text language models, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # surpressing some transformers' output

import spacy
import string
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from alibi.explainers import AnchorText
from alibi.datasets import fetch_movie_sentiment
from alibi.utils import spacy_model
from alibi.utils import DistilbertBaseUncased, BertBaseUncased, RobertaBase
```

### Load movie review dataset

The `fetch_movie_sentiment` function returns a `Bunch` object containing the features, the targets and the target names for the dataset.

```python
movies = fetch_movie_sentiment()
movies.keys()
```

```
dict_keys(['data', 'target', 'target_names'])
```

```python
data = movies.data
labels = movies.target
target_names = movies.target_names
```

Define shuffled training, validation and test set

```python
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)
```

### Apply CountVectorizer to training set

```python
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
```

```
CountVectorizer()
```

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.CountVectorizer

```
CountVectorizer()
```

### Fit model

```python
np.random.seed(0)
clf = LogisticRegression(solver='liblinear')
clf.fit(vectorizer.transform(train), train_labels)
```

```
LogisticRegression(solver='liblinear')
```

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.\
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.LogisticRegression

```
LogisticRegression(solver='liblinear')
```

### Define prediction function

```python
predict_fn = lambda x: clf.predict(vectorizer.transform(x))
```

### Make predictions on train and test sets

```python
preds_train = predict_fn(train)
preds_val = predict_fn(val)
preds_test = predict_fn(test)
print('Train accuracy: %.3f' % accuracy_score(train_labels, preds_train))
print('Validation accuracy: %.3f' % accuracy_score(val_labels, preds_val))
print('Test accuracy: %.3f' % accuracy_score(test_labels, preds_test))
```

```
Train accuracy: 0.980
Validation accuracy: 0.754
Test accuracy: 0.759
```

### Load spaCy model

English multi-task CNN trained on OntoNotes, with GloVe vectors trained on Common Crawl. Assigns word vectors, context-specific token vectors, POS tags, dependency parse and named entities.

```python
model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)
```

### Instance to be explained

```python
class_names = movies.target_names

# select instance to be explained
text = data[4]
print("* Text: %s" % text)

# compute class prediction
pred = class_names[predict_fn([text])[0]]
alternative =  class_names[1 - predict_fn([text])[0]]
print("* Prediction: %s" % pred)
```

```
* Text: a visually flashy but narratively opaque and emotionally vapid exercise in style and mystification .
* Prediction: negative
```

### Initialize anchor text explainer with `unknown` sampling

* `sampling_strategy='unknown'` means we will perturb examples by replacing words with UNKs.

```python
explainer = AnchorText(
    predictor=predict_fn, 
    sampling_strategy='unknown',
    nlp=nlp,
)
```

### Explanation

```python
explanation = explainer.explain(text, threshold=0.95)
```

Let us now take a look at the anchor. The word `flashy` basically guarantees a negative prediction.

```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

```
Anchor: flashy
Precision: 0.99

Examples where anchor applies and model predicts negative:
a UNK flashy UNK UNK opaque and emotionally vapid exercise in style UNK mystification .
a UNK flashy UNK UNK UNK and emotionally UNK exercise UNK UNK and UNK UNK
a UNK flashy UNK narratively opaque UNK UNK UNK exercise in style and UNK UNK
UNK visually flashy UNK narratively UNK and emotionally UNK UNK UNK UNK UNK mystification .
UNK UNK flashy UNK UNK opaque and emotionally UNK UNK in UNK and UNK .
a visually flashy but UNK UNK and UNK UNK UNK in style UNK mystification .
a visually flashy but UNK opaque UNK emotionally vapid UNK in UNK and mystification .
a UNK flashy but narratively UNK UNK emotionally vapid exercise in style UNK mystification UNK
a UNK flashy but narratively opaque UNK emotionally vapid exercise in style and mystification .
a visually flashy UNK UNK opaque UNK UNK UNK exercise in UNK UNK UNK .

Examples where anchor applies and model predicts positive:
UNK UNK flashy but narratively UNK and UNK UNK UNK in style and UNK UNK
```

### Initialize anchor text explainer with word `similarity` sampling

Let's try this with another perturbation distribution, namely one that replaces words by similar words instead of UNKs.

```python
explainer = AnchorText(
    predictor=predict_fn, 
    sampling_strategy='similarity',   # replace masked words by simialar words
    nlp=nlp,                          # spacy object
    sample_proba=0.5,                 # probability of a word to be masked and replace by as similar word
)
```

```python
explanation = explainer.explain(text, threshold=0.95)
```

The anchor now shows that we need more to guarantee the negative prediction:

```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

```
Anchor: exercise AND vapid
Precision: 0.99

Examples where anchor applies and model predicts negative:
that visually flashy but tragically opaque and emotionally vapid exercise under genre and glorification .
another provably flashy but hysterically bulky and emotionally vapid exercise arounds style and authorization .
that- visually flashy but narratively opaque and politically vapid exercise in style and mystification .
a unintentionally decal but narratively thick and emotionally vapid exercise in unflattering and mystification .
the purposely flashy but narratively rosy and emotionally vapid exercise in style and mystification .
thievery intentionally flashy but hysterically gray and anally vapid exercise in style and mystification .
a irrationally flashy but narratively smoothness and purposefully vapid exercise near style and diction .
a medio flashy but narratively blue and economically vapid exercise since style and intuition .
a visually flashy but narratively opaque and anally vapid exercise onwards style and mystification .
each purposefully flashy but narratively gorgeous and emotionally vapid exercise in style and mystification .

Examples where anchor applies and model predicts positive:
a visually punchy but tragically opaque and hysterically vapid exercise in minimalist and mystification .
a visually discernible but realistically posh and physically vapid exercise around style and determination .
```

We can make the token perturbation distribution sample words that are more similar to the ground truth word via the `top_n` argument. Smaller values (default=100) should result in sentences that are more coherent and thus more in the distribution of natural language which could influence the returned anchor. By setting the `use_proba` to True, the sampling distribution for perturbed tokens is proportional to the similarity score between the possible perturbations and the original word. We can also put more weight on similar words via the `temperature` argument. Lower values of `temperature` increase the sampling weight of more similar words. The following example will perturb tokens in the original sentence with probability equal to `sample_proba`. The sampling distribution for the perturbed tokens is proportional to the similarity score between the ground truth word and each of the `top_n` words.

```python
explainer = AnchorText(
    predictor=predict_fn, 
    sampling_strategy='similarity',  # replace masked words by simialar words
    nlp=nlp,                         # spacy object
    use_proba=True,                  # sample according to the similiary distribution
    sample_proba=0.5,                # probability of a word to be masked and replace by as similar word
    top_n=20,                        # consider only top 20 words most similar words
    temperature=0.2                  # higher temperature implies more randomness when sampling
)
```

```python
explanation = explainer.explain(text, threshold=0.95)
```

```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

```
Anchor: exercise AND flashy
Precision: 1.00

Examples where anchor applies and model predicts negative:
a visually flashy but sarcastically brown and reflexively vapid exercise between style and mystification .
this visually flashy but intentionally shiny and emotionally vapid exercise in style and appropriation .
a visually flashy but narratively glossy and critically vapid exercise in accentuate and omission .
a visually flashy but historically glossy and purposely rapid exercise within stylesheet and equivocation .
each visually flashy but intently opaque and emotionally quickie exercise throughout style and mystification .
that reflexively flashy but narratively opaque and romantically melodramatic exercise within style and mystification .
a equally flashy but narratively boxy and emotionally predictable exercise in classism and exaggeration .
a visually flashy but narratively opaque and emotionally vapid exercise between style and mystification .
a visually flashy but emphatically opaque and emotionally vapid exercise walkthrough classism and mystification .
a verbally flashy but sarcastically opaque and emotionally dramatic exercise in design and mystification .

Examples where anchor applies and model predicts positive:
that visually flashy but narratively boxy and reflexively insignificant exercise outside minimalist and appropriation .
```

### Initialize language model

Because the Language Model is computationally demanding, we can run it on the GPU. Note that this is optional, and we can run the explainer on a non-GPU machine too.

```python
# the code runs for non-GPU machines too
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```

We provide support for three transformer-based language models: `DistilbertBaseUncased`, `BertBaseUncased`, and `RobertaBase`. We initialize the language model as follows:

```python
# language_model = RobertaBase()
# language_model = BertBaseUncased()
language_model = DistilbertBaseUncased()
```

```
Some layers from the model checkpoint at distilbert-base-uncased were not used when initializing TFDistilBertForMaskedLM: ['activation_13']
- This IS expected if you are initializing TFDistilBertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFDistilBertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
All the layers of TFDistilBertForMaskedLM were initialized from the model checkpoint at distilbert-base-uncased.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFDistilBertForMaskedLM for predictions without further training.
```

### Initialize anchor text explainer with `language_model` sampling (`parallel` filling)

* `sampling_strategy='language_model'` means that the words will be sampled according to the output distribution predicted by the language model
* `filling='parallel'` means the only one forward pass is performed. The words are the sampled independently of one another.

```python
# initialize explainer
explainer = AnchorText(
    predictor=predict_fn,
    sampling_strategy="language_model",   # use language model to predict the masked words
    language_model=language_model,        # language model to be used
    filling="parallel",                   # just one pass through the transformer
    sample_proba=0.5,                     # probability of masking a word
    frac_mask_templates=0.1,              # fraction of masking templates (smaller value -> faster, less diverse)
    use_proba=True,                       # use words distribution when sampling (if False sample uniform)
    top_n=20,                             # consider the fist 20 most likely words
    temperature=1.0,                      # higher temperature implies more randomness when sampling
    stopwords=['and', 'a', 'but', 'in'],  # those words will not be sampled
    batch_size_lm=32,                     # language model maximum batch size
)
```

```python
explanation = explainer.explain(text, threshold=0.95)
```

```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

```
Anchor: exercise AND flashy AND opaque
Precision: 0.95

Examples where anchor applies and model predicts negative:
a visually flashy but visually opaque and politically photographic exercise in style and mystification.
a visually flashy but visually opaque and emotionally expressive exercise in style and mystification.
a visually flashy but visually opaque and socially visual exercise in style and mystification.
a visually flashy but ultimately opaque and visually dramatic exercise in style and mystification.
a visually flashy but often opaque and highly conscious exercise in style and mystification.
a visually flashy but historically opaque and intensely thorough exercise in style and mystification.
a visually flashy but socially opaque and visually an exercise in style and mystification.
a visually flashy but emotionally opaque and socially an exercise in style and mystification.
a visually flashy but emotionally opaque and visually creative exercise in style and mystification.
a visually flashy but sometimes opaque and subtly photographic exercise in style and mystification.

Examples where anchor applies and model predicts positive:
a visually flashy but visually opaque and deeply enjoyable exercise in imagination and mystification.
a visually flashy but somewhat opaque and highly an exercise in reflection and mystification.
a visually flashy but narratively opaque and deeply imaginative exercise in style and mystification.
a visually flashy but narratively opaque and visually challenging exercise in style and mystification.
a visually flashy but narratively opaque and intensely challenging exercise in style and mystification.
a surprisingly flashy but narratively opaque and highly rigorous exercise in style and mystification.
a very flashy but narratively opaque and highly imaginative exercise in style and mystification.
```

### Initialize anchor text explainer with `language_model` sampling (`autoregressive` filling)

* `filling='autoregressive'` means that the words are sampled one at the time (autoregressive). Thus, following words to be predicted will be conditioned one the previously generated words.
* `frac_mask_templates=1` in this mode (overwriting it with any other value will not be considered).
* **This procedure is computationally expensive**.

```python
# initialize explainer
explainer = AnchorText(
    predictor=predict_fn,
    sampling_strategy="language_model",  # use language model to predict the masked words
    language_model=language_model,       # language model to be used
    filling="autoregressive",            # just one pass through the transformer
    sample_proba=0.5,                    # probability of masking a word
    use_proba=True,                      # use words distribution when sampling (if False sample uniform)
    top_n=20,                            # consider the fist 20 most likely words
    stopwords=['and', 'a', 'but', 'in']  # those words will not be sampled
)
```

```python
explanation = explainer.explain(text, threshold=0.95, batch_size=10, coverage_samples=100)
```

```python
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('\nExamples where anchor applies and model predicts %s:' % pred)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_true']]))
print('\nExamples where anchor applies and model predicts %s:' % alternative)
print('\n'.join([x for x in explanation.raw['examples'][-1]['covered_false']]))
```

```
Anchor: flashy AND exercise AND vapid
Precision: 0.96

Examples where anchor applies and model predicts negative:
a visually flashy but emotionally opaque and emotionally vapid exercise in mystery and mystification.
a slightly flashy but narratively opaque and deliberately vapid exercise in style and detail.
a visually flashy but narratively accessible and emotionally vapid exercise in style and creativity.
a fairly flashy but narratively vivid and emotionally vapid exercise in style and mystification.
a somewhat flashy but socially opaque and emotionally vapid exercise in style and technique.
a fairly flashy but extremely opaque and emotionally vapid exercise in beauty and mystification.
a little flashy but extremely lively and fairly vapid exercise in relaxation and mystification.
a slightly flashy but highly opaque and somewhat vapid exercise in movement and breathing.
a visually flashy but narratively sensitive and emotionally vapid exercise in style and mystification.
a visually flashy but emotionally opaque and emotionally vapid exercise in beauty and mystery.

Examples where anchor applies and model predicts positive:
```

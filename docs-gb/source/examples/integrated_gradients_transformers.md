# Integrated gradients for transformers models

In this example, we apply the integrated gradients method to two different sentiment analysis models. The first one is a pretrained sentiment analysis model from the [transformers](https://github.com/huggingface/transformers) library. The second model is a combination of a pretrained (distil)BERT model and a simple feed forward network. The entire model, **(distil)BERT** and feed forward network, is trained on the **IMDB reviews** dataset.

In text classification models, **integrated gradients (IG)** define an attribution value for each word in the input sentence. The attributions are calculated considering the integral of the model gradients with respect to the word embedding layer along a straight path from a baseline instance $x^\prime$ to the input instance $x.$ A description of the method can be found [here](https://docs.seldon.io/projects/alibi/en/stable/methods/IntegratedGradients.html). Integrated gradients was originally proposed in Sundararajan et al., ["Axiomatic Attribution for Deep Networks"](https://arxiv.org/abs/1703.01365)

Note

To enable support for IntegratedGradients, you may need to run

```bash
pip install alibi[tensorflow]
```

```python
import re
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm

from tqdm import tqdm
from typing import Optional, Union, List, Dict, Tuple
from IPython.display import HTML

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from transformers.optimization_tf import WarmUp
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer

from alibi.explainers import IntegratedGradients
```

Here we define some functions needed to process the data and visualize. For consistency with other [text examples](https://github.com/ramonpzg/alibi/blob/rp-alibi-newdocs-dec23/doc/source/examples/integrated_gradients_imdb.ipynb) in alibi, we will use the **IMDB reviews** dataset provided by Keras. Since the dataset consists of reviews that are already tokenized, we need to decode each sentence and re-convert them into tokens using the **(distil)BERT** tokenizer.

```python
def decode_sentence(x: List[int], reverse_index: Dict[int, str], unk_token: str = '[UNK]') -> str:
    """ 
    Decodes the tokenized sentences from keras IMDB dataset into plain text.
    
    Parameters
    ----------
    x
        List of integers to be docoded.
    revese_index
        Reverse index map, from `int` to `str`.
    unk_token
        Unkown token to be used.
        
    Returns
    -------
        Decoded sentence.
    """
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, unk_token) for i in x])


def process_sentences(sentence: List[str], 
                      tokenizer: PreTrainedTokenizer, 
                      max_len: int) -> Dict[str, np.ndarray]:
    """
    Tokenize the text sentences.
    
    Parameters
    ----------
    sentence
        Sentence to be processed.
    tokenizer
        Tokenizer to be used.
    max_len
        Controls the maximum length to use by one of the truncation/padding parameters.
    
    
    Returns
    -------
    Tokenized representation containing:
     - input_ids
     - attention_mask
    """
    # since we are using the model for classification, we need to include special char (i.e, '[CLS]', ''[SEP]')
    # check the example here: https://huggingface.co/transformers/v4.4.2/quicktour.html
    z = tokenizer(sentence, 
                  add_special_tokens=True, 
                  padding='max_length', 
                  max_length=max_len, 
                  truncation=True,
                  return_attention_mask = True,  
                  return_tensors='np')
    return z

def process_input(sentence: List[str],
                  tokenizer: PreTrainedTokenizer,
                  max_len: int) -> Tuple[np.ndarray, dict]:
    """
    Preprocess input sentence befor sending to transformer model.
    
    Parameters
    -----------
    sentence
        Sentence to be processed.
    tokenizer
        Tokenizer to be used.
    max_len
        Controls the maximum length to use by one of the truncation/padding parameters.
        
    Returns
    -------
    Tuple consisting of the input_ids and a dictionary contaning the attention_mask.
    """
    # tokenize the sentences using the transformer's tokenizer.
    tokenized_samples = process_sentences(sentence, tokenizer, max_len)
    X_test = tokenized_samples['input_ids'].astype(np.int32)

    # the values of the kwargs have to be `tf.Tensor`. 
    # see transformers issue #14404: https://github.com/huggingface/transformers/issues/14404
    # solved from v4.16.0
    kwargs = {k: tf.constant(v) for k, v in tokenized_samples.items() if k != 'input_ids'}
    return X_test, kwargs
```

```python
def  hlstr(string: str , color: str = 'white') -> str:
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"


def colorize(attrs: np.ndarray, cmap: str = 'PiYG') -> List:
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    
    Parameters
    ----------
    attrs
        Attributions to be visualized.
    cmap
        Matplotlib cmap type.
    """
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)
    return list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))


def display(X: np.ndarray, 
            attrs: np.ndarray, 
            tokenizer: PreTrainedTokenizer,
            pred: np.ndarray) -> None:
    """
    Display the attribution of a given instance.
    
    Parameters
    ----------
    X
        Instance to display the attributions for.
    attrs
        Attributions values for the given instance.
    tokenizer
        Tokenizer to be used for decoding.
    pred
        Classification label (prediction) for the given instance.
    """
    pred_dict = {1: 'Positive review', 0: 'Negative review'}
    
    # remove padding
    fst_pad_indices = np.where(X ==tokenizer.pad_token_id)[0]
    if len(fst_pad_indices) > 0:
        X, attrs = X[:fst_pad_indices[0]], attrs[:fst_pad_indices[0]]
    
    # decode tokens and get colors
    tokens = [tokenizer.decode([X[i]]) for i in range(len(X))]
    colors = colorize(attrs)
    
    print(f'Predicted label =  {pred}: {pred_dict[pred]}')
    return HTML("".join(list(map(hlstr, tokens, colors))))
```

## Automodel

In this section, we will use the Tensorflow auto model for sequence classification provided by the [transformers](https://github.com/huggingface/transformers) library.

The model is pretrained on the [Stanford Sentiment Treebank (SST)](https://huggingface.co/datasets/sst) dataset. The **Stanford Sentiment Treebank** is the first corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language.

Each phrase is labeled as either negative, somewhat negative, neutral, somewhat positive or positive. The corpus with all 5 labels is referred to as **SST-5** or **SST fine-grained**. Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as **SST-2** or **SST binary**. In this example, we will use a text classifier pretrained on the **SST-2** dataset.

```python
# load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
auto_model_distilbert = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```
All model checkpoint layers were used when initializing TFDistilBertForSequenceClassification.

All the layers of TFDistilBertForSequenceClassification were initialized from the model checkpoint at distilbert-base-uncased-finetuned-sst-2-english.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFDistilBertForSequenceClassification for predictions without further training.
```

The `auto_model_distilbert` output is a custom object containing the output logits. We use a wrapper to transform the output into a tensor and apply a softmax function to the logits.

```python
class AutoModelWrapper(keras.Model):
    def __init__(self, transformer: keras.Model, **kwargs):
        """
        Constructor.
        
        Parameters
        ----------
        transformer
            Transformer to be wrapped.
        """
        super().__init__()
        self.transformer = transformer

    def call(self, 
             input_ids: Union[np.ndarray, tf.Tensor], 
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             training: bool = False):
        """
        Performs forward pass throguh the model.
        
        Parameters
        ----------
        input_ids
            Indices of input sequence tokens in the vocabulary.
        attention_mask
            Mask to avoid performing attention on padding token indices.
        
        Returns
        -------
        Classification probabilities.
        """
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, training=training)
        return tf.nn.softmax(out.logits, axis=-1)
    
    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
```

```python
auto_model = AutoModelWrapper(auto_model_distilbert)
```

### Calculate integrated gradients

The auto model consists of a main **distilBERT** layer (layer 0) followed by two dense layers.

```python
auto_model.layers[0].layers
```

```
[<transformers.models.distilbert.modeling_tf_distilbert.TFDistilBertMainLayer at 0x7f4cb5435e80>,
 <keras.layers.core.dense.Dense at 0x7f4ca8058c10>,
 <keras.layers.core.dense.Dense at 0x7f4ca8058e80>,
 <keras.layers.core.dropout.Dropout at 0x7f4ca01db1f0>]
```

We will proceed with the embedding layer from **distilBERT**. We calculate attributions to the outputs of the **embedding layer** for which we can easily construct an appropriate baseline for the **IG** by replacing the regular tokens with the \[PAD] token (i.e. a neutral token) and keeping the other special tokens (e.g. \[CLS], \[SEP], \[UNK], \[PAD]). By including special tokens such as \[CLS], \[SEP], \[UNK], we ensure that the attribution for those tokens will be 0 if we use the embedding layer. The 0 attribution is due to integration between $\[x, x]$ which is 0. Note that if we considered a hidden layer instead, we would inevitably capture higher order interaction between the input tokens. Moreover, the embedding layer is our first choice since we cannot compute attributions for the raw input due to its discrete structure (i.e., we cannot differentiate the output of the model with respect to the discrete input representation). **That being said, you can use any other layer and compute attributions to the outputs of it instead.**

```python
#  Extracting the embeddings layer
layer = auto_model.layers[0].layers[0].embeddings

# # Extract the first layer from the transformer
# layer = auto_model.layers[0].layers[0].transformer.layer[0]
```

```python
n_steps = 50
internal_batch_size = 5
method = "gausslegendre"

# define Integrated Gradients explainer
ig  = IntegratedGradients(auto_model,
                          layer=layer,
                          n_steps=n_steps, 
                          method=method,
                          internal_batch_size=internal_batch_size)
```

Here we consider some simple sentences such as "I love you, I like you", "I love you, I like you, but I also kind of dislike you" .

```python
# define some text to be explained
text_samples = [
    'I love you, I like you', 
    'I love you, I like you, but I also kind of dislike you',
    'Everything is so nice about you'
]

# process input to be explained
X_test, kwargs = process_input(sentence=text_samples,
                               tokenizer=tokenizer,
                               max_len=256)
```

```python
# get predictions
predictions = auto_model(X_test, **kwargs).numpy().argmax(axis=1)

# get the baseline
mask = np.isin(X_test, tokenizer.all_special_ids)
baselines = X_test * mask + tokenizer.pad_token_id * (1 - mask)

# get explanation
explanation = ig.explain(X_test, 
                         forward_kwargs=kwargs,
                         baselines=baselines, 
                         target=predictions)
```

Let's check the attributions' shapes.

```python
# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)
```

```
Attributions shape: (3, 256, 768)
```

As you can see, the attribution of each token corresponds to a tensor of `768` elements. We compress all this information into a single number buy summing up all `768` components. The nice thing about this is that we still remain consistent with the **Completeness Axiom**, which states that the attributions add up to the difference between the output of our model for the given instance and the output of our model for the given baseline.

```python
attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)
```

```
Attributions shape: (3, 256)
```

```python
index = 1
display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)
```

```
Predicted label =  0: Negative review
```

\[CLS] i love you , i like you , but i also kind of dislike you \[SEP]

**Note that since the sentence is classified as negative, words like `dislike` contribute positively to the score while words like `love` contribute negatively.**

## Sentiment analysis on IMDB with fine-tuned model head.

### Load and process data

```python
# constants
max_features = 10000

# load imdb reviews datasets.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# remove the first integer token which is a special character that marks the beginning of the sentence
x_train = [x[1:] for x in x_train]
x_test = [x[1:] for x in x_test]

# get mappings. The keys are transformed to lower case since we will use uncased models.
reverse_index = {value: key.lower() for (key, value) in imdb.get_word_index().items()}
```

### Load model and corresponding tokenizer

Now we have to load the model and the corresponding tokenizer. You can chose between the **BERT** model or the **distilBERT** model. Note that we will be finetuning those models which will require access to a **GPU**. In our experiments, we trained distilBERT on a single **Quadro RTX 5000** which requires around **5GB** of memory. The entire training took around **5-6 min**. We recommend using **distilBERT** as it is lighter and we did not noticed a big difference in performance between the two models after finetuning.

```python
# choose whether to use the BERT or distilBERT model by selecting the appropriate name
model_name = 'distilbert-base-uncased'
# model_name = 'bert-base-uncased'
```

```python
# load model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# define maximum input length
max_len = 256

if model_name == 'bert-base-uncased':
    # training parameters: https://huggingface.co/fabriceyhc/bert-base-uncased-imdb
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 8
    num_warmup_steps = 1546
    num_train_steps = 15468
    power = 1.0
    
elif model_name == 'distilbert-base-uncased':
    # training parameters: https://huggingface.co/lvwerra/distilbert-imdb
    init_lr = 5e-05
    min_lr_ratio = 0
    batch_size = 16
    num_warmup_steps = 0
    num_train_steps = int(np.ceil(len(x_train) / batch_size)) 
    power = 1.0
    
else:
    raise ValueError('Unknown model name.')
```

```
Some layers from the model checkpoint at distilbert-base-uncased were not used when initializing TFDistilBertForSequenceClassification: ['vocab_transform', 'vocab_projector', 'activation_13', 'vocab_layer_norm']
- This IS expected if you are initializing TFDistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFDistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some layers of TFDistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['dropout_39', 'pre_classifier', 'classifier']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

Decoding each sentence in the **Keras IMDB** tokenized dataset to obtain the corresponding plain text. The dataset is already in a pretty good shape, so we don't need to do extra preprocessing. The only thing that we do is to replace the unknown tokens with the appropriate tokenizer's unknown token (i.e., `[UNK]`)

```python
X_train, X_test = [], []

# decode training sentences
for i in range(len(x_train)):
    tr_sentence = decode_sentence(x_train[i], reverse_index, unk_token=tokenizer.unk_token)
    X_train.append(tr_sentence)

# decode testing sentences
for i in range(len(x_test)):
    te_sentence = decode_sentence(x_test[i], reverse_index, unk_token=tokenizer.unk_token)
    X_test.append(te_sentence)
```

Retokenizing the plain text using the **(distil)BERT** tokenizer.

```python
# tokenize datasets
X_train = process_sentences(X_train, tokenizer, max_len)
X_test = process_sentences(X_test, tokenizer, max_len)
```

Construct the Tensorflow datasets for training and testing.

```python
train_ds = tf.data.Dataset.from_tensor_slices(((*X_train.values() ,), y_train))
train_ds = train_ds.shuffle(1024).batch(batch_size).repeat()

test_ds = tf.data.Dataset.from_tensor_slices(((*X_test.values(), ), y_test))
test_ds = test_ds.batch(batch_size)
```

### Train model

Here we train a classification model by leveraging the pretrained **(distil)BERT** transformer.

```python
filepath = './model_transformers/'  # change to desired save directory
checkpoint_path = os.path.join(filepath, model_name)
load_model = False

# define linear learning schedules
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=init_lr,
    decay_steps=num_train_steps - num_warmup_steps,
    end_learning_rate=init_lr * min_lr_ratio,
    power=power,
)

# include learning rate warmup
if num_warmup_steps:
    lr_schedule = WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=lr_schedule,
        warmup_steps=num_warmup_steps,
    )

if not load_model:
    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    
    # fit and save the model
    model.fit(x=train_ds, validation_data=test_ds, steps_per_epoch=num_train_steps)
    model.save_pretrained(checkpoint_path)
else:
    # load and compile the model
    model = model.from_pretrained(checkpoint_path)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )
    
    # evaluate the model
    model.evaluate(test_ds)
```

```
1563/1563 [==============================] - 459s 291ms/step - loss: 0.2721 - sparse_categorical_accuracy: 0.8859 - val_loss: 0.2141 - val_sparse_categorical_accuracy: 0.9142
```

```python
# wrap the finetuned model
auto_model = AutoModelWrapper(model)
```

### Calculate integrated gradients

We pick the first 10 sentences from the test set as examples. You can easily add some of your text here too, as we exemplify it.

```python
# include IMDB reviews from the test dataset
text_samples = [decode_sentence(x_test[i], reverse_index, unk_token=tokenizer.unk_token) for i in range(10)]

# inlcude your text here
text_samples.append("best movie i've ever seen nothing bad to say about it")

# process input before passing it to the explainer
X_test, kwargs = process_input(sentence=text_samples,
                               tokenizer=tokenizer,
                               max_len=max_len)
```

We calculate the attributions with respect to the first embedding layer of the **(distil)BERT**. You can choose any other layer.

```python
if model_name == 'bert-base-uncased':
    layer = auto_model.layers[0].layers[0].embeddings
    # layer = auto_model.layers[0].layers[0].encoder.layer[2]

elif model_name == 'distilbert-base-uncased':
    layer = auto_model.layers[0].layers[0].embeddings
    # layer = auto_model.layers[0].layers[0].transformer.layer[0]

else:
    raise ValueError('Unknown model name.')
```

```python
n_steps = 50
method = "gausslegendre"
internal_batch_size = 5

# define Integrated Gradients explainer
ig  = IntegratedGradients(auto_model,
                          layer=layer,
                          n_steps=n_steps, 
                          method=method,
                          internal_batch_size=internal_batch_size)
```

```python
# compute model's prediction and construct baselines
predictions = auto_model(X_test, **kwargs).numpy().argmax(axis=1)

# construct the baseline as before
mask = np.isin(X_test, tokenizer.all_special_ids)
baselines = X_test * mask + tokenizer.pad_token_id * (1 - mask)

# get explanation
explanation = ig.explain(X_test, 
                         forward_kwargs=kwargs,
                         baselines=baselines, 
                         target=predictions)
```

```python
# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape:', attrs.shape)
```

```
Attributions shape: (11, 256, 768)
```

```python
attrs = attrs.sum(axis=2)
print('Attributions shape:', attrs.shape)
```

```
Attributions shape: (11, 256)
```

### Check attributions for our example

```python
index = -1
display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)
```

```
Predicted label =  1: Positive review
```

\[CLS] best movie i ' ve ever seen nothing bad to say about it \[SEP]

### Check attribution for some test examples

```python
index = 0
display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)
```

```
Predicted label =  0: Negative review
```

\[CLS] please give this one a miss br br \[UNK] \[UNK] and the rest of the cast rendered terrible performances the show is flat flat flat br br i don ' t know how michael madison could have allowed this one on his plate he almost seemed to know this wasn ' t going to work out and his performance was quite \[UNK] so all you madison fans give this a miss \[SEP]

```python
index = 1
display(X=X_test[index], attrs=attrs[index], pred=predictions[index], tokenizer=tokenizer)
```

```
Predicted label =  1: Positive review
```

\[CLS] this film requires a lot of patience because it focuses on mood and character development the plot is very simple and many of the scenes take place on the same set in frances \[UNK] the sandy dennis character apartment but the film builds to a disturbing climax br br the characters create an atmosphere \[UNK] with sexual tension and psychological \[UNK] it ' s very interesting that robert alt ##man directed this considering the style and structure of his other films still the trademark alt ##man audio style is evident here and there i think what really makes this film work is the brilliant performance by sandy dennis it ' s definitely one of her darker characters but she plays it so perfectly and convincing ##ly that it ' s scary michael burns does a good job as the mute young man regular alt ##man player michael murphy has a small part the \[UNK] moody set fits the content of the story very well in short this movie is a powerful study of loneliness sexual \[UNK] and desperation be patient \[UNK] up the atmosphere and pay attention to the wonderful ##ly written script br br i praise robert alt ##man this is one of his many films that deals with unconventional fascinating subject matter this film is disturbing but it ' s sincere and it ' s sure to \[UNK] a strong emotional response from the viewer if you want to see an unusual film some might even say bizarre this is worth the \[SEP]

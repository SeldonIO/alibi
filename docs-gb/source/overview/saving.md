# Saving and loading

Alibi includes experimental support for saving and loading a subset of explainers to disk using the `dill` module.

To save an explainer, simply call the `save` method and provide a path to a directory (a new one will be created if it doesn't exist):
```python
explainer.save('path')
```

Alibi doesn't save the model/prediction function that is passed into the explainer so when loading the explainer you will need to provide it again:
```python
from alibi.saving import load_explainer
explainer = load_explainer('path', predictor=predictor)
```

## Details and limitations
Every explainer will save the following artifacts as a minimum:
```bash
path/meta.dill
path/explainer.dill
```
Here `meta.dill` is the metadata of the explainer (including the Alibi version used to create it) and `explainer.dill` is the serialized explainer. Some explainers may save more artifacts, e.g. `AnchorText` additionally saves `path/nlp` which is the `spacy` model used to initialize the explainer using the native `spacy` saving functionality (`pickle` based) whilst `AnchorImage` also saves the custom Python segmentation function under `path/segmentation_fn.dill`.

When loading a saved explainer, a warning will be issued if the runtime Alibi version is different from the version used to save the explainer. **It is highly recommended to use the same Alibi, Python and dependency versions as were used to save the explainer to avoid potential bugs and incompatibilities**.
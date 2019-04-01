import spacy
from spacy.util import get_package_path
from spacy.cli import download, link


def spacy_model(model: str = 'en_core_web_md') -> None:
    """
    Download spaCy model.

    Parameters
    ----------
    model
        Model to be downloaded
    """
    try:
        spacy.load(model)
    except OSError:
        download(model)

        # https://github.com/explosion/spaCy/issues/3435
        package_path = get_package_path(model)
        link(model, model, force=True, model_path=package_path)

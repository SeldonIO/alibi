import spacy
import subprocess


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
        cmd = ['python', '-m', 'spacy', 'download', model]
        subprocess.run(cmd)

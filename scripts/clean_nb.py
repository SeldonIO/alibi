import os
import glob
from pathlib import Path
import nbformat

NOTEBOOK_DIR = 'doc/source/examples'
ALL_NOTEBOOKS = {Path(x).name for x in glob.glob(str(Path(NOTEBOOK_DIR).joinpath('*.ipynb')))}


def clean_notebooks():
    for notebook_name in ALL_NOTEBOOKS:
        nb = nbformat.read(os.path.join(NOTEBOOK_DIR, notebook_name), as_version=4)
        for cell in nb.cells:
            for clean_fn in (
                    remove_pycharm_metadata,
                    ):
                clean_fn(cell)
        nbformat.write(nb, os.path.join(NOTEBOOK_DIR, notebook_name))


def remove_pycharm_metadata(cell):
    if hasattr(cell, 'metadata') and hasattr(cell.metadata, 'pycharm'):
        del cell.metadata['pycharm']


if __name__ == '__main__':
    clean_notebooks()

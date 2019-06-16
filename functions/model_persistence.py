from joblib import dump, load
from sklearn.base import BaseEstimator
import os


def dump_clf(clf: BaseEstimator, path: str = 'configuration/', name: str = 'clf') -> None:
    """
    Write a BaseClassifier to file for later reuse.

    :param clf: sklearn classifier
    :param path: relative path to folder where the model should be saved. Defaults to config folder
    :param name: desired filename for the file including extension (joblib)
    """
    path = os.path.join(os.getcwd(), path)
    assert os.path.isdir(path), 'Specified path doesn\'t exists.'

    dump(clf, os.path.join(path, name))


def load_clf(filepath: str) -> BaseEstimator:
    """
    Load a classifier from a joylib file

    :param filepath: relative path to the file, including extension
    :return: the sklearn classifier
    """
    path = os.path.join(os.getcwd(), filepath)
    assert os.path.isfile(path), 'Specified file doesn\'t exists'

    return load(path)

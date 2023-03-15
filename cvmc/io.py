from typing import Union, Any, Tuple
from collections import OrderedDict
import os

import pickle

from flax.core import freeze
from flax.serialization import to_state_dict

from .utils.types import PyTree


def save_model(path: str, params: PyTree, **extra) -> None:

    """Save a model to disk.

    Parameters
    ----------
    path : str
        Path to save the model to.
    params : PyTree
        Model parameters.
    """

    if "###state_dict###" in extra:
        raise ValueError(
            'Additional arrays cannot be named "###state_dict###" as the name is used internally!'
        )
    elif "###extra###" in extra:
        raise ValueError(
            'Additional arrays cannot be named "###extra###" as the name is used internally!'
        )

    path = os.path.abspath(path)
    path, ext = os.path.splitext(path)

    if ext not in [".pkl", ".pickle"]:
        ext = ".pkl"

    path += ext

    state_dict = to_state_dict(params)
    to_save = {"###state_dict###": state_dict, "###extra###": extra}

    with open(path, "wb") as file:
        pickle.dump(to_save, file)


def load_model(path: str) -> Union[PyTree, Tuple[PyTree, Any]]:

    """Load a model from disk.

    Parameters
    ----------
    path : str
        Path to load the model from.

    Returns
    -------
    Union[PyTree, Any]
        The model parameters followed by any additional data.

    """

    path = os.path.abspath(path)

    with open(path, "rb") as file:
        data = pickle.load(file)

    params = freeze(data["###state_dict###"])
    rest = OrderedDict(data["###extra###"])

    return (params, rest) if rest else params

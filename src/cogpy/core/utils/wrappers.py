import numpy as np
from copy import deepcopy, copy
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import xarray as xr
import logging
import os
from typing import Any, Optional, Protocol, cast
from matplotlib.axes import Axes
from functools import partial


def get_wrapper(attr_name):
    def _get_wrapper(method):
        def wrap_method(self, *args, get=False, **kwargs):
            temp_self = copy(self)
            value = method(temp_self, *args, **kwargs)
            if get:
                return value

            else:
                setattr(self, attr_name, value)

        return wrap_method

    return _get_wrapper


def dropnames(method):
    def wrap_method(self, *args, **kwargs):
        self.A = self.asarray()
        method(self, *args, **kwargs)

    return wrap_method


def flatten_arg(input, arg):
    output = [deepcopy(input) for i in range(len(input[arg]))]
    for i_elem, o_elem in zip(input[arg], output):
        o_elem[arg] = i_elem

    return output


def kwarg_intlist_wrapper(key_word):
    def _wrapper(func):
        """
        func: int -> object

        wrapped_func: list(int) -> list(object)
        """

        def wrapped_func(*args, **kwargs):
            if isinstance(kwargs[key_word], int):
                return [func(*args, **kwargs)]

            flat_kwarg = flatten_arg(kwargs, key_word)
            flat_output = np.empty_like(kwargs[key_word], object)
            for i, wrap_kwarg in enumerate(flat_kwarg):
                if isinstance(wrap_kwarg[key_word], int):
                    flat_output[i] = func(*args, **wrap_kwarg)

            return flat_output

        return wrapped_func

    return _wrapper


def add_docs_for(other_func):
    def dec(func):
        func.__doc__ = other_func.__doc__ + "\n\n" + func.__doc__
        return func

    return dec


def partial_method(applicator, method):
    func = partial(applicator, method=method)
    func.__doc__ = (
        applicator.__doc__
        + """

        Method
        ______

        """
        + method.__doc__
    )
    return func


# The wrapped function must accept an Axes kwarg named `ax`
class _PlotFunc(Protocol):
    def __call__(self, *args: Any, ax: Axes, **kwargs: Any) -> Any: ...


def ax_plot(plot_func: _PlotFunc) -> _PlotFunc:
    def _plot_func(*args: Any, ax: Optional[Axes] = None, **kwargs: Any) -> Any:
        if ax is None:
            ax = plt.gca()
        return plot_func(*args, ax=ax, **kwargs)

    return cast(_PlotFunc, _plot_func)


def timeit_decorator(func):
    logger = logging.getLogger(__name__)

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(
            f"'{func.__name__}' executed in {end_time - start_time:.5f} seconds"
        )
        return result

    return wrapper


@contextmanager
def execution_timer(description="Time taken", indent_level=0):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    indent = "\t" * indent_level
    logger.info(f"{indent}{description}: {elapsed_time:.5f} seconds")


def safe_save(func, dst_file, *args, **kwargs):
    """Wrapper to safely execute saving functions."""
    logger = logging.getLogger(__name__)
    try:
        func(dst_file, *args, **kwargs)
        logger.info(f"Successfully saved to {dst_file}")
    except Exception as e:
        logger.error(f"Could not save to {dst_file}")
        logger.error(str(e))


@contextmanager
def temp_open(path, mode):
    # if this fails there is nothing left to do anyways
    file = open(path, mode)

    try:
        yield file
    finally:
        file.close()
        os.remove(path)

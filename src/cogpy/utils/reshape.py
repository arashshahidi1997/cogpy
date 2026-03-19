import numpy as np
from ._functools import num_to_tuple


def roll_dim(x, nroll):
    """
    reshapes the array x, for example if x.shape = (3,6,9,0)
    roll_dim(x,-1).shape = (6,9,0,3)
    Parameters
    ----------
    x: array (arbitrary dimensions)
    nroll: int

    Returns
    -------
    reshaped version of x with dimesnions rolled by `nroll` times.
    """
    return x.transpose(np.roll(np.arange(x.ndim), nroll))


def reshape_axes(arr, axes, shape):
    """
    Reshape the specified axes of an array while preserving the column order.

    Args:
        arr (ndarray): Input array.
        axes (int): axes to reshape.
        shape (tuple): Desired shape after reshaping.

    Returns:
        ndarray: Reshaped array.

    Example:
        original_shape = (30, 64, 20, 10)
        arr = np.random.rand(*original_shape)
        reshaped_array = reshape_axes(arr, axes=1, shape=(8, 8))
        assert (reshaped_array.reshape(-1) == arr.reshape(-1)).all()
        print(reshaped_array.shape)  # (30, 8, 8, 20, 10)

    Note:
    Does not support axes=-1

    """
    shape = num_to_tuple(shape)
    axes = num_to_tuple(axes)
    axes = tuple(
        [ax % len(arr.shape) for ax in axes]
    )  # convert negative axes to positive
    X = flush_axes(arr, axes)
    shape_up_to_axes = X.shape[: -len(axes)]
    X = X.reshape(shape_up_to_axes + shape)
    X = unflush_axes(X, num_axes=len(shape), dst_axis=axes[0])
    return X


def ravel_dims(arr, axis1, axis2):
    """
    Ravel the specified dimensions of an array.

    Args:
        arr (ndarray): Input array.
        axis1 (int): First axis to ravel.
        axis2 (int): Second axis to ravel.

    Returns:
        ndarray: Raveled array.

    Example:
    a = np.random.rand(16, 10, 100)
    raveled = ravel_dims(a, 0, 1)
    assert raveled.shape == (160, 100)
    """
    return reshape_axes(
        arr, (axis1, axis2), np.prod(np.array(arr.shape)[[axis1, axis2]])
    )


def unflush_axes(X, num_axes, dst_axis):
    """
    Move the last `num_axes` axes of an array `X` to a new position specified by `dst_axes`.

    Args:
        X (ndarray): Input array.
        num_axes (int): Number of axes to move.
        dst_axis (int): Destination starting axis where the axes will be moved.

    Returns:
        ndarray: Array with the last `num_axes` axes moved to the new position.

    Example:
        X = np.random.rand(10, 20, 30, 40, 50)
        moved_array = unflush_axes(X, num_axes=3, dst_axes=1)
        print(moved_array.shape)  # (10, 30, 40, 50, 20)
    """
    last_axes = last_naxes(num_axes)
    dst_axes = tuple(dst_axis + np.arange(num_axes))
    X = np.moveaxis(X, last_axes, dst_axes)
    return X


def flush_axes(X, src_axes):
    """
    Move the axes specified by `src_axes` axes of an array `X` to the last axes.

    Args:
        X (ndarray): Input array.
        src_axes (int): Source axes where the axes will be moved to end.

    Returns:
        ndarray: Array with the `src_axes` moved to the end axes.

    Example:
        X = np.random.rand(10, 20, 30, 40, 50)
        moved_array = flush_axes(X, src_axes=(1,2))
        print(moved_array.shape)  # (10, 40, 50, 20, 30)
    """
    src_axes = num_to_tuple(src_axes)
    num_axes_to_move = len(src_axes)
    dst_axes = last_naxes(num_axes_to_move)
    X = np.moveaxis(X, src_axes, dst_axes)
    return X


def last_naxes(num_axes):
    shape_range = np.arange(num_axes)
    last_axes = tuple(-(shape_range + 1)[::-1])
    return last_axes

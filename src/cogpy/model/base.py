import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from abc import ABC, abstractmethod

class BaseMode(ABC):
    """
    An abstract base class representing a generic mode shape in a domain.

    Attributes:
        shape (tuple of ints): The dimensions of the domain.
        boundary_condition (str): The type of boundary condition applied to the domain.

    Methods:
        generate_mode(sigma): Abstract method to generate a mode based on given parameters.
        shift_mode(mode, loc): Abstract method to shift a mode to a new location within the domain.
    """
    def __init__(self, shape, boundary_condition='non-periodic'):
        """
        Initializes the BaseMode with given shape and boundary condition.

        Parameters:
            shape (tuple of ints): The shape of the domain.
            boundary_condition (str): The type of boundary condition (e.g., 'periodic', 'non-periodic').
        """
        self.shape = shape
        self.boundary_condition = boundary_condition

    @abstractmethod
    def generate_mode(self, sigma):
        """
        Generates a mode based on given parameters.

        Parameters:
            sigma (list or array of floats): Parameters defining the mode shape or distribution characteristics.

        Returns:
            An implementation-defined representation of the mode.
        """
        pass

    @abstractmethod
    def shift_mode(self, mode, loc):
        """
        Shifts a given mode to a specified location within the domain.

        Parameters:
            mode: The mode to shift, format and type are implementation-dependent.
            loc (tuple of ints): The new location to shift the mode to.

        Returns:
            The shifted mode, format and type are implementation-dependent.
        """
        pass

class BaseCover(ABC):
    """
    An abstract base class for covering a domain with modes.

    Attributes:
        shape (tuple of ints): The dimensions of the domain.
        boundary_condition (str): The type of boundary condition applied to the domain.
    """
    def __init__(self, shape, boundary_condition='non-periodic'):
        """
        Initializes the BaseCover with given shape and boundary condition.

        Parameters:
            shape (tuple of ints): The shape of the domain.
            boundary_condition (str): The type of boundary condition (e.g., 'periodic', 'non-periodic').
        """
        self.shape = shape
        self.boundary_condition = boundary_condition

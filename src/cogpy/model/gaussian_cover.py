import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal
import itertools
from .base import BaseCover, BaseMode

class GaussianMode(BaseMode):
    """
    A class that represents Gaussian modes within a domain, inheriting from BaseMode.

    This class overrides the abstract methods of BaseMode to generate and shift Gaussian modes.
    """
    def __init__(self, shape, boundary_condition='non-periodic'):
        super().__init__(shape, boundary_condition)

    def generate_mode(self, cov_matrix=None, sigma=None):
        """
        Generates a Gaussian mode based on the given covariance matrix or standard deviations.

        Parameters:
            cov_matrix (2D array-like): The covariance matrix defining the Gaussian distribution.
            sigma (list of floats, optional): The standard deviations for each dimension, defining the width of the Gaussian distribution. Ignored if cov_matrix is provided.
            center (list of floats, optional): The center coordinates for the Gaussian distribution. Defaults to the center of the domain.

        Returns:
            xr.DataArray: An xarray DataArray representing the Gaussian mode across the specified dimensions.
        """
        grids = [np.linspace(-dim//2, dim//2-1, dim) for dim in self.shape]
        mesh = np.meshgrid(*grids, indexing='ij')
        pos = np.stack(mesh, axis=-1)
        
        if cov_matrix is None:
            cov_matrix = np.diag(np.array(sigma)**2)
        else:
            cov_matrix = np.array(cov_matrix)

        if sigma is not None:
            sigma = np.array(sigma)
            if sigma.shape != (len(self.shape),):
                raise ValueError("Sigma must have the same number of dimensions as the mode.")
            cov_matrix = np.diag(sigma**2)

        rv = multivariate_normal([0]*len(self.shape), cov_matrix)

        dims = ["dim_{}".format(i) for i in range(len(self.shape))]
        coords = {dim_name: grid for dim_name, grid in zip(dims, grids)}
        return xr.DataArray(rv.pdf(pos), coords=coords, dims=dims)

    def shift_mode(self, mode: xr.DataArray, loc=None):
        """
        Shifts a Gaussian mode to a new location within the domain, respecting the boundary conditions.

        Parameters:
            mode (xr.DataArray): The mode to shift.
            loc (tuple of ints): The location to shift the mode to.

        Returns:
            xr.DataArray: The shifted Gaussian mode.
        """
        if loc is None:
            loc = self.center

        if self.boundary_condition == "periodic":
            shifts = [(l + dim) % dim for l, dim in zip(loc, self.shape)]
            shifted_mode = mode.roll(**{dim_name: shift for dim_name, shift in zip(mode.dims, shifts)})

        else:  # non-periodic
            shifted_mode = mode.shift(**{dim_name: locus for dim_name, locus in zip(mode.dims, loc)}, fill_value=0)
        return shifted_mode
    
class GaussianCover(BaseCover):
    """
    A class that covers a domain with Gaussian modes, inheriting from BaseCover.

    This class utilizes Gaussian modes to fill the domain based on specified parameters.

    Attributes:
        cov_matrix (2D array-like): The covariance matrix for the Gaussian modes.
        sigma (list of floats): The standard deviations for the Gaussian modes.
        spacing (tuple of ints): The spacing between the centers of adjacent Gaussian modes.
    """
    def __init__(self, shape, cov_matrix=None, sigma=None, spacing=None, boundary_condition='non-periodic'):
        """
        Initializes the GaussianCover with the domain shape, mode characteristics, and boundary condition.

        Parameters:
            shape (tuple of ints): The shape of the domain.
            cov_matrix (2D array-like, optional): The covariance matrix for the Gaussian modes. Defaults to None.
            sigma (list of floats, optional): The standard deviations for the Gaussian modes. Ignored if cov_matrix is provided. Defaults to None.
            spacing (tuple of ints): The spacing between the centers of adjacent Gaussian modes.
            boundary_condition (str, optional): The type of boundary condition. Defaults to 'non-periodic'.
        """
        super().__init__(shape, boundary_condition)
        self.cov_matrix = cov_matrix
        self.sigma = sigma
        self.spacing = spacing
        if spacing is None:
            self.spacing = tuple([s//4 for s in shape])
            assert all(s > 0 for s in self.spacing), "Spacing must be positive in all dimensions."
            assert len(self.spacing) == len(shape), "Spacing must have the same number of dimensions as shape."

        self.gaussian_mode = GaussianMode(shape, boundary_condition=self.boundary_condition)
        self.canonical_mode = self.gaussian_mode.generate_mode(cov_matrix=self.cov_matrix, sigma=self.sigma)
        self.canonical_min = self.canonical_mode.min()
        self.canonical_max = self.canonical_mode.max()

        self.indices = [np.arange(-dim//2 + s//2, dim//2, s) for dim, s in zip(shape, spacing)]
        self.combinations = list(itertools.product(*self.indices))

        self.modes = [self.gaussian_mode.shift_mode(self.canonical_mode, loc) for loc in self.combinations]

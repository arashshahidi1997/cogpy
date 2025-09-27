"""
module for identifying local minima and maxima
"""
import numpy as np
from scipy.ndimage import filters
import scipy.ndimage as nd
import pandas as pd

def detect_extrema(x, footprint, minima=True):
    """
    Detects extrema (minima or maxima) in an array.

    Args:
        x (ndarray): The input array.
        footprint (ndarray): The footprint or structuring element for the local maximum filter.
        minima (bool, optional): Specifies whether to detect minima or maxima.
                                 Defaults to True (detect minima).

    Returns:
        ndarray: Boolean mask of the detected extrema.

    """
    _filt = filters.minimum_filter if minima else filters.maximum_filter
    return _filt(x, footprint=footprint) == x


class Extrema:
    """
    Detects troughs in a grid signal using the local maximum filter.

    Args:
        grid_signal (GridSignal): The input grid signal.
        minima (bool, optional): Specifies whether to detect minima or maxima.
                                 Defaults to True (detect minima).
        der_step (int, optional): The step size for generating the derivative kernel.
                                  Defaults to 1.
        axes (int, optional): The axis along which to detect troughs.
                              Defaults to -1.

    Attributes:
        der_kernel (ndarray): The derivative kernel used for detecting troughs.
        ext_arr (ndarray): Boolean mask of the troughs.
        df (DataFrame): DataFrame containing the coordinates and values of the detected troughs.
        empty (DataFrame): Empty DataFrame with the same columns as df.
        gs (GridSignal): Owner grid_signal.

    """

    def __init__(self, grid_signal, minima=True, der_step=1, axes=-1):
        """
        Initializes the Extrema object.

        Args:
            grid_signal (GridSignal): The input grid signal.
            minima (bool, optional): Specifies whether to detect minima or maxima.
                                     Defaults to True (detect minima).
            der_step (int, optional): The step size for generating the derivative kernel.
                                      Defaults to 1.
            axes (int, optional): The axis along which to detect troughs.
                                  Defaults to -1.
        """
        # Read array
        arr = np.moveaxis(np.asarray(grid_signal), axes, 0)  # Axis 0 = time

        # Construct footprint
        struct = nd.generate_binary_structure(2, 1)
        footprint = nd.iterate_structure(struct, der_step).astype(int)
        self.der_kernel = np.expand_dims(footprint, 0)  # Expand dims in the time direction

        # Extrema detection
        self.ext_arr = detect_extrema(arr, footprint=self.der_kernel, minima=minima)
        nonzero_indices = self.ext_arr.nonzero()
        df = pd.DataFrame(np.argwhere(self.ext_arr), columns=['t', 'h', 'w'])  # Coordinates of extrema
        df['val'] = pd.Series(arr[nonzero_indices])  # Values of the signal at non-zero indices

        # Sort DataFrame
        self.df = df.sort_values(['t', 'h', 'w'])

        # Empty DataFrame
        self.empty = pd.DataFrame(columns=self.df.columns)

        # Owner grid_signal
        self.gridshape = grid_signal.gridshape
        # self.gs = grid_signal

    def __getitem__(self, t):
        """
        Returns the DataFrame containing the troughs for a specific time.

        Args:
            t (int): The time index.

        Returns:
            DataFrame: DataFrame with the troughs for the specified time.

        """
        try:
            return self.df[self.df['t'] == t]
        except:
            return self.empty

    def get_wave(self, clu):
        """
        Returns the DataFrame containing the troughs for a specific cluster.

        Args:
            clu (int): The cluster label.

        Returns:
            DataFrame: DataFrame with the troughs for the specified cluster.

        """
        return self.df[self.df['Clu'] == clu]

    def channel_column(self):
        """
        Adds a column representing the channel index to the DataFrame.

        """
        self.df['ch'] = pd.Series(np.ravel_multi_index((self.df['h'], self.df['w']), dims=self.gridshape))

    def detect_waves(self, propagate_radius=1):
        """
        Performs wave propagation and clustering based on the detected extrema.

        Args:
            propagate_radius (int, optional): Radius for wave propagation.
                                             Defaults to 1.

        """
        propagate_size = 2 * propagate_radius + 1
        self.propagator_kernel = np.ones((3, propagate_size, propagate_size))
        not_ext_arr = np.invert(self.ext_arr)
        labels = self.ext_arr.astype(float)
        labels[self.ext_arr.nonzero()] = np.arange(1, 1 + len(self.df))
        labels[labels == 0] = np.inf

        # Frame by frame wave propagation
        clusters = np.copy(labels)

        for i in range(clusters.shape[0] - 1):
            clusters[i:i + 2] = nd.filters.minimum_filter(clusters[i:i + 2], footprint=self.propagator_kernel)
            clusters[i:i + 2][(not_ext_arr[i:i + 2]).nonzero()] = np.inf

        clu = clusters[tuple(self.df[['t', 'h', 'w']].to_numpy().T)].astype(int)

        # Relabel clusters
        x = np.zeros(np.max(clu) + 1, dtype=int)
        uclu = np.unique(clu)
        x[uclu] = np.arange(len(uclu))
        clu = x[clu]

        # Add cluster id to DataFrame
        self.df['Clu'] = clu

    def get_waves(self, interval):
        """
        Returns the DataFrame containing the troughs within a specific interval.

        Args:
            interval (tuple): Start and end time of the interval.

        """
        pass

    def center_of_mass(self):
        """
        Calculates the center of mass for each cluster.

        Returns:
            DataFrame: DataFrame with the cluster's center of mass coordinates, average value, and cluster label.

        """
        df = self.df[self.df == 0]
        clusters = np.unique(df['Clu'])
        df_com_list = []
        for i, c in enumerate(clusters):
            h_mean, w_mean, val_mean = df[df['Clu'] == c][['h', 'w', 'val']].mean()
            df_com_list.append([h_mean, w_mean, val_mean, c])

        df_com = pd.DataFrame(df_com_list, columns=['h', 'w', 'val', 'Clu'])
        return df_com

# Batch
class Waves:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.w = df.groupy('Clu')
        
    def duration(self):
        # return self.w.apply(duration)
        pass

    

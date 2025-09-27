# static TLS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from typing import Optional, cast
import warnings
import numpy as np
import scipy.ndimage as nd
import pandas as pd
from matplotlib.axes import Axes
from kneed import KneeLocator
from ..utils.wrappers import ax_plot
from .channel_feature import ChannelFeatures 

EPSILON = 0.000001

class OutlierDetector(ClusterMixin, BaseEstimator):
    """
    detects outliers from given features
    """
    kn: KneeLocator
    k_dist: np.ndarray
    dbscan: DBSCAN

    # Pre-fit params
    min_samples: int
    eps: Optional[float]

    # Post-fit attrs (non-optional after fit)
    eps_: float
    labels_: np.ndarray
    labels_raw: np.ndarray

    def __init__(self, min_samples=15, eps=None, eps_optimize_k=10):
        self.min_samples = min_samples
        self.eps = eps
        self.eps_optimize_k = eps_optimize_k

    def optimize_eps(self, X, k):
        assert not np.isnan(X).any(), "KneeLocator: Input data contains NaN values"
        self.k_dist = calculate_k_distance(X, k=k)
        self.kn = KneeLocator(np.arange(1,len(self.k_dist)+1), self.k_dist, curve='convex', direction='increasing')

    def fit(self, X, y=None):
        if self.eps is None:
            self.optimize_eps(X, k=self.eps_optimize_k)
        if self.kn.knee_y is None:
            warnings.warn(
                f"KneeLocator could not find a knee in k-distance curve; "
                f"setting eps to fallback EPSILON={EPSILON}",
                RuntimeWarning
            )
            self.eps = EPSILON
        else:
            self.eps = float(self.kn.knee_y)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan.fit(X)
        self.labels_raw = self.dbscan.labels_
        self.labels_ = (self.dbscan.labels_ == -1)
        self.labels = self.labels_
        return self
    
    def fit_predict(self, X, y=None):
        self.fit(X, y=y)
        return self.labels_

    @ax_plot
    def plot_k_distances(self,  ax: Optional[Axes] = None):
        ax = cast(Axes, ax)
        ax.plot(self.k_dist)
        ax.axvline(self.kn.knee, color='k', linestyle='--')
        ax.axhline(self.kn.knee_y, color='k', linestyle='--')
        eps_str = f'$\epsilon={self.kn.knee_y:.2f}$'
        ax.set(xlabel='k', ylabel='$\epsilon$', title=f'k-distance plot\n{eps_str}')

    @ax_plot
    def plot_k_distances(self, ax: None) -> None:
        ax.plot(self.k_dist)

        # Dot access is fine because self.kn exists after fit
        knee_x = self.kn.knee
        knee_y = self.kn.knee_y if self.kn.knee_y is not None else EPSILON
        if self.kn.knee_y is None:
            warnings.warn(f"No knee found; using EPSILON={EPSILON}", RuntimeWarning)

        if knee_x is not None:
            ax.axvline(knee_x, color="k", linestyle="--")
        ax.axhline(knee_y, color="k", linestyle="--")

        ax.set(xlabel="k", ylabel="$\\epsilon$", title=f"k-distance plot\n$\\epsilon={knee_y:.2f}$")


DetectBadsPipe = Pipeline([
    ('features', ChannelFeatures()),
    ('scaler', StandardScaler()),
    ('dbscan', OutlierDetector())
], memory='cache', verbose=True)

class DetectBads(BaseEstimator):
    # Declare attributes so linters know they exist after fit()
    pipe: Pipeline
    feature_df: Optional[pd.DataFrame]  # or keep as Any if you prefer
    labels_: Optional[np.ndarray]
    labels_raw: Optional[np.ndarray]

    def __init__(self) -> None:            
        self.pipe = Pipeline([
            ('features', ChannelFeatures()),
            ('scaler', StandardScaler()),
            ('dbscan', OutlierDetector())
        ], memory='cache', verbose=True)

        self.feature_df = None
        self.labels_ = None
        self.labels_raw = None

    def fit(self, X, y=None):
        self.pipe.fit(X)

        self.features_: ChannelFeatures = self.pipe.named_steps['features']
        db = cast(OutlierDetector, self.pipe.named_steps['dbscan'])

        # attrs come from the correct step, not the pipeline
        self.feature_df = getattr(self.features_, 'feature_df', None)

        self.labels_ = db.labels_
        self.labels_raw = db.labels_raw
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        # mypy/pylance: we asserted labels_ exists post-fit
        return cast(np.ndarray, self.labels_)

    def predict(self, X):
        # Access steps via named_steps and cast for type checkers
        feats = self.pipe.named_steps['features']
        scaler = self.pipe.named_steps['scaler']
        db = cast(OutlierDetector, self.pipe.named_steps['dbscan'])

        # Make static analyzers happy and fail early if not fitted
        check_is_fitted(db, 'eps_')
        eps: float = db.eps_

        Z = scaler.transform(feats.transform(X))
        # Refit a DBSCAN with the fitted eps/min_samples for new data
        return DBSCAN(eps=eps, min_samples=db.min_samples).fit_predict(Z) == -1

# example usage
# dbads = DetectBads.fit(fb_sigx)
# DetectBads.fit_transform(X, y=None)

# %% optimize dbscan epsilon

# Calculate the k-distance graph
def calculate_k_distance(data, k):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    k_distances = np.sort(distances[:, k], axis=0)
    k_distances = nd.gaussian_filter1d(k_distances, sigma=3)
    return k_distances


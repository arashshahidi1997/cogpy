from stppg import StdDiffusionKernel, HawkesLam, MarkedSpatialTemporalPointProcess
from utils import plot_spatio_temporal_points, plot_spatial_intensity

np.random.seed(0)
np.set_printoptions(suppress=True)

# parameters initialization
mu     = .1
kernel = StdDiffusionKernel(C=1., beta=1., sigma_x=.1, sigma_y=.1)
lam    = HawkesLam(mu, kernel, maximum=1e+3)
pp     = SpatialTemporalPointProcess(lam)

# generate points
points, sizes = pp.generate(
    T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
    batch_size=500, verbose=True)

# plot intensity of the process over the time
plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
    t_slots=1000, grid_size=50, interval=50)
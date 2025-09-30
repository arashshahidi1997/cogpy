import numpy as np
import matplotlib.pyplot as plt

# Subgrid representatives
def subgrid_representatives(area_shape, gridshape=(16,16)):
    grid = np.arange(np.prod(gridshape)).reshape(gridshape)
    # define the size of each representative area
    subgrid_shape = (gridshape[0] // area_shape[0], gridshape[1] // area_shape[1])
    # split the grid into 4 x 4 sub-grids
    sub_grids = np.split(grid, subgrid_shape[0], axes=0)
    sub_grids = [np.split(sub_grid, subgrid_shape[1], axes=1) for sub_grid in sub_grids]
    # choose the representative channel for each sub-grid
    representatives = np.zeros(subgrid_shape, dtype=int)
    for (i,j), rep in np.ndenumerate(representatives):
        representatives[i, j] = sub_grids[i][j].mean().round().astype(int)
    return representatives, grid

def plot_subgrid(grid, representatives):
    plt.imshow(grid, cmap='gray')
    # add markers at the coordinates of the representative channels
    for rep in representatives:
        x, y = np.argwhere(grid == rep)[0]
        plt.plot(x, y, 'ro')
    plt.show()

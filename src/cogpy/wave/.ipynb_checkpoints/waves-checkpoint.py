# waves
"""
module for wave calculations
"""
import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy.ndimage as nd

def duration(wave):
    return wave.t.max() - wave.t.min() + 1

def trajectory(lext_df):
    return lext_df.set_index('Clu').apply(list, axis=1).\
            to_frame().reset_index().rename(columns={0:'coo'}).\
            groupby('Clu').agg({'coo':np.array})

def peak(wave):
    return

def relext(wave):
    return

def remove_close(wave):
    pass

def laplace(wave):
    nd.filters.laplace()
    pass

def convexity():
    pass

def phase_coherence():
    pass

def contour(wave):
    pass

def split_waves(wave):
    waves = []
    return waves

def drop_short_waves(wave):
    pass

def eccentricity(df):
    pass

def wavelet(df):
    pass

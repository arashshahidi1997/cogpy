import numpy as np
import pandas as pd

def wave_gen(x: pd.Series, wave_df: pd.DataFrame):
    """
    x: xr.DataArray
    wave_df: pd.DataFrame
    """
    for ion, ioff in zip(wave_df.ion, wave_df.ioff):
        yield x[ion:ioff]

# """FROM CUSTOM PACKAGES"""

# """load imports"""
# from ...preprocessing.bad_channel import interpolate
# from ...io import *
# from ...datasets.load import (load_sample,
#                              load_raw_sample
#                             )

# """grid signal imports"""
# from ...preprocessing import filt
# from ...wave import extrema

# """grid plot imports"""
# from ...plot import frame_plot as fplt
# fr = fplt.FramePlot

# """wrappers and decorators"""
# from .. import wrappers as wp
# from ..mypath import *

# __all__ = ['ld','lds', 'GridSignal',
#            'to_batch', 'to_frame','filt',
#            'interpolate','extrema','fplt','fr',
#            'wp','load_sample','load_sample_batch',
#            'load_raw_sample', 'DATA_DIR', 'LABBOX_DIR',
#            'RESULT_DIR', 'TEST_DIR', 'SRC_DIR'
#            ]
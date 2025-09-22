import os
from pathlib import Path
from ..io.load_session import load_session
from ..signal.grid_signal import GridBatch

def load_sample():
    module_path = os.path.dirname(__file__)
    sample_session = Path(module_path)/'cached_data/sample_signal'
    return load_session(sample_session)

def load_sample_batch():
    module_path = os.path.dirname(__file__)
    sample_files = Path(module_path+'/cached_data/').glob('*.dat')
    
    sig_batch = []
    for session in sample_files:
        sig_batch.append(load_session(session))

    return GridBatch(sig_batch)


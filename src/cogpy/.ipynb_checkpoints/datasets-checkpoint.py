from src.io_deprecated.load_session import load_session
from src.mypath import SRC_DIR
sample_session = SRC_DIR/'data/sample_signal'

def load_sample():
    return load_session(sample_session)

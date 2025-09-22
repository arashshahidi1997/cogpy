from pathlib import Path

BASE_DIR = Path.cwd()
while BASE_DIR.name != 'LFP-Grid':
    BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR/'data'
LABBOX_DIR = Path('/storage/share/matlab/labbox/')
RESULT_DIR = BASE_DIR/'results'
TEST_DIR = BASE_DIR/'tests'
SRC_DIR = BASE_DIR/'src'
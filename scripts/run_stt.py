"""Import torch before Sherpa so CUDA shared libraries are loaded."""
import runpy
import sys
from pathlib import Path

import torch  # noqa: F401

root = Path(__file__).resolve().parents[1]
examples = root / 'upstream' / 'python-api-examples'
sys.path.insert(0, str(examples))
runpy.run_path(str(examples / 'streaming_server.py'), run_name='__main__')


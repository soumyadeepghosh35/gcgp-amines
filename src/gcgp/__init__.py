"""Shared code for the GC-GP amine property notebooks.

The notebooks in ``notebooks/`` import everything they need from this package,
so the modelling logic lives in one place and every notebook runs the same code.
"""

import os

# Hide TensorFlow's informational and warning messages (for example the
# "Cholesky decomposition was not successful" warning). Numerical failures are
# still caught by the code and reported in the fold and run tables.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "1.1.0"

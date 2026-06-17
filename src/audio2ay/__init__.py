"""audio2ay — instrumental audio → AY-3-8910 register stream (50 Hz default)."""

import os as _os

# PyTorch (Demucs) and TensorFlow (Basic Pitch) each ship their own OpenMP
# runtime. When both are loaded in one process on Windows the duplicate Intel
# OpenMP (libiomp5md.dll) intermittently corrupts memory and the interpreter
# dies with a native ACCESS_VIOLATION (exit code 0xC0000005) — no traceback,
# no output file. These guards must be set before numpy/torch/tensorflow are
# imported, so they live at the top of the package __init__ (the first module
# loaded). Force these values because conflicting pre-set values can still lead
# to intermittent ACCESS_VIOLATION crashes on Windows when TF + torch coexist.
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

__version__ = "0.1.0"

# Hardware target (Vector-06C). Override via AY_CLOCK env or pass clock= to APIs.
AY_CLOCK_HZ = 1_500_000
FRAME_RATE_HZ = 50  # Classic YM rate; CLI can opt into 100 Hz updates.
SAMPLES_PER_FRAME_44K = 44_100 // FRAME_RATE_HZ  # 882

"""audio2ay — instrumental audio → AY-3-8910 register stream (50 Hz default)."""

import os as _os

# Safety guard in case any indirect dep ships a second OpenMP runtime alongside
# PyTorch. KMP_DUPLICATE_LIB_OK=TRUE prevents an ACCESS_VIOLATION on Windows
# when two libiomp5md.dll copies end up loaded in the same process.
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "0.1.0"

# Hardware target (Vector-06C). Override via AY_CLOCK env or pass clock= to APIs.
AY_CLOCK_HZ = 1_500_000
FRAME_RATE_HZ = 50  # Classic YM rate; CLI can opt into 100 Hz updates.
SAMPLES_PER_FRAME_44K = 44_100 // FRAME_RATE_HZ  # 882

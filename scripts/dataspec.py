"""Central path configuration for the Biohub cell-tracking datasets."""

import os
import sys
from pathlib import Path

INTERACTIVE = sys.stderr.isatty()

USERNAME = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))

_BASE = Path(__file__).resolve().parent.parent

# Kaggle competition mount (present when running in a Kaggle notebook).
_KAGGLE_TRAIN = Path(
    "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
)


def _default_dataset_path() -> Path:
    """Resolve the dataset directory.

    Priority:
      1. ``$VOLT_DATA_DIR`` — explicit override (any environment).
      2. the Kaggle competition ``train`` mount, when running on Kaggle.
      3. local ``./data/dense_channel`` — default for local development.

    Scripts also accept ``--data-dir`` to override this per run (e.g. to point
    at the competition ``test`` directory for prediction).
    """
    env = os.environ.get("VOLT_DATA_DIR")
    if env:
        return Path(env)
    if _KAGGLE_TRAIN.exists():
        return _KAGGLE_TRAIN
    return _BASE / "data/train"


DATASET_PATH = _default_dataset_path()

# Directory that *contains* train/ and test/, i.e. one level above the train
# split. The downloader writes both splits under here and the EDA scripts read
# both from here, so it has to be one value rather than a constant repeated in
# each script. Correct in all three environments because DATASET_PATH always
# ends in the train split: <repo>/data/train, $VOLT_DATA_DIR (set to
# .../biohub/train), or the Kaggle competition mount's train/.
DATA_ROOT = DATASET_PATH.parent

PREDICTIONS_PATH = _BASE / "predictions"
RESULTS_PATH = _BASE / "results"
WEIGHTS_PATH = _BASE / "weights"
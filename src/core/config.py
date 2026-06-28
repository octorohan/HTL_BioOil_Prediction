from pathlib import Path

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"

MODEL_DIR = OUTPUT_DIR / "models"

# -------------------------------------------------------
# Dataset
# -------------------------------------------------------

DATASET_PATH = DATA_DIR / "bbb2637-sup-0001-datas1.xlsx"

# -------------------------------------------------------
# Random State
# -------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.30

# -------------------------------------------------------
# Target
# -------------------------------------------------------

TARGET_COLUMN = "Oil"

# -------------------------------------------------------
# Columns removed from training
# -------------------------------------------------------

DROP_COLUMNS = [
    "Oil",
    "Biocrude",
    "Gas",
    "Char",
    "WaterPhase",
    "Author",
    "Ref",
]
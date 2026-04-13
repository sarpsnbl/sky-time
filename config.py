"""
config.py
=========
Central configuration for Time-of-Day Estimation — ConvNeXt backbones only.
Adjust these parameters before running main.py or tune.py.
"""


class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR  = "dataset_512"
    IMAGE_SIZE = 224

    # --- Cross-Validation -----------------------------------------------------
    FOLD            = 0     # Which CV fold to use (0-indexed); ignored when TRAIN_ALL_FOLDS=True
    N_SPLITS        = 5
    VAL_RATIO       = 0.2
    TRAIN_ALL_FOLDS = True

    # --- Image Heuristic Features ---------------------------------------------
    USE_IMAGE_FEATURES = True

    # --- Model ----------------------------------------------------------------
    # Supported: "convnext_tiny" | "convnext_small"
    MODEL        = "convnext_small"
    PRETRAINED   = True
    FREEZE_UNTIL = "features.4"
    HIDDEN_DIM   = 384
    DROPOUT      = 0.167786

    # --- Training -------------------------------------------------------------
    EPOCHS         = 75
    UNFREEZE_EPOCH = 25         # Epoch to unfreeze full backbone (None = never)
    BATCH_SIZE     = 16
    LR             = 0.001896
    ETA_MIN        = 2.781919e-07
    WEIGHT_DECAY   = 0.000241
    NUM_WORKERS    = 8
    USE_AMP        = True
    WEIGHTED_SAMPLER = True

    # --- Augmentation ---------------------------------------------------------
    # ConvNeXt works best without augmentation; kept for experimentation.
    AUG_MAGNITUDE   = "light"    # "none" | "light" | "medium" | "heavy"
    MIXUP_ALPHA     = 0.184888
    LABEL_NOISE_STD = 0.042620

    # --- Test-Time Augmentation -----------------------------------------------
    TTA_ENABLED = False
    TTA_FLIPS   = 2

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR = "checkpoints"
    CHECKPOINT = None
    EVAL_ONLY  = False
    SEED       = 42

    # --- Optuna Hyperparameter Optimisation -----------------------------------
    OPTUNA_N_TRIALS         = 30
    OPTUNA_EPOCHS           = 30
    OPTUNA_TIMEOUT_SECONDS  = None
    OPTUNA_N_STARTUP_TRIALS = 8
    OPTUNA_CV_FOLDS         = 1
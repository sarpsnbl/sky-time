"""
config.py
=========
Central configuration for Time-of-Day Estimation — ConvNeXt backbones only.
Adjust these parameters before running main.py or tune.py.
"""

class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR  = "dataset_512"
    IMAGE_SIZE = 512

    # --- Cross-Validation -----------------------------------------------------
    FOLD            = 0     
    N_SPLITS        = 5
    VAL_RATIO       = 0.2
    TRAIN_ALL_FOLDS = True

    # --- Image Heuristic Features ---------------------------------------------
    USE_IMAGE_FEATURES = True

    # --- Model ----------------------------------------------------------------
    MODEL        = "convnext_tiny"
    PRETRAINED   = True
    FREEZE_UNTIL = "features.4"
    HIDDEN_DIM   = 384
    DROPOUT      = 0.1555

    # --- Training & Hardware Optimizations ------------------------------------
    EPOCHS           = 80
    UNFREEZE_EPOCH   = None         
    BATCH_SIZE       = 4
    ACCUM_STEPS      = 2       # Effective batch size = BATCH_SIZE * ACCUM_STEPS = 8
    
    # The "Free Lunches"
    USE_AMP           = True   # Mixed Precision
    USE_COMPILE       = True   # torch.compile() for graph optimization (PT 2.0+)
    USE_CHANNELS_LAST = True   # NHWC memory format for Tensor Core speedup
    USE_8BIT_OPTIM    = True   # bitsandbytes 8-bit AdamW to save VRAM
    
    LR               = 4.42e-04
    ETA_MIN          = 1.83e-05
    WEIGHT_DECAY     = 0.0373
    NUM_WORKERS      = 8
    WEIGHTED_SAMPLER = True

    # --- Augmentation ---------------------------------------------------------
    AUG_MAGNITUDE   = "heavy"
    MIXUP_ALPHA     = 0.1776
    LABEL_NOISE_STD = 0.0437

    # --- Test-Time Augmentation -----------------------------------------------
    TTA_ENABLED = False
    TTA_FLIPS   = 2

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR = "checkpoints"
    CHECKPOINT = None
    EVAL_ONLY  = False
    SEED       = 42

    # --- Optuna Hyperparameter Optimisation -----------------------------------
    OPTUNA_N_TRIALS         = 15
    OPTUNA_EPOCHS           = 60
    OPTUNA_TIMEOUT_SECONDS  = None
    OPTUNA_N_STARTUP_TRIALS = 4
    OPTUNA_CV_FOLDS         = 1
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
    MODEL        = "swin_t"
    PRETRAINED   = True
    FREEZE_UNTIL = "features.4"
    HIDDEN_DIM   = 384
    DROPOUT      = 0.029274015233555814

    # --- Training & Hardware Optimizations ------------------------------------
    EPOCHS           = 80
    UNFREEZE_EPOCH   = None         
    BATCH_SIZE       = 8
    ACCUM_STEPS      = 1      # Effective batch size = BATCH_SIZE * ACCUM_STEPS
    
    # The "Free Lunches"
    USE_AMP           = True   # Mixed Precision
    USE_COMPILE       = True   # torch.compile() for graph optimization (PT 2.0+)
    USE_CHANNELS_LAST = True   # NHWC memory format for Tensor Core speedup
    USE_8BIT_OPTIM    = True   # bitsandbytes 8-bit AdamW to save VRAM
    
    LR               = 1.46e-04
    ETA_MIN          = 3.77e-06
    WEIGHT_DECAY     = 4.32e-03
    NUM_WORKERS      = 8
    WEIGHTED_SAMPLER = True

    # --- Augmentation ---------------------------------------------------------
    AUG_MAGNITUDE   = "moderate"
    MIXUP_ALPHA     = 0.14411297399224515
    LABEL_NOISE_STD = 0.03142830830721105

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
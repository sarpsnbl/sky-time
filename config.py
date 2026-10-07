"""
config.py
=========
Central configuration for Time-of-Day Estimation
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
    FREEZE_UNTIL = "features.6"
    HIDDEN_DIM   = 384
    USE_FILM     = True    # Feature-wise Linear Modulation for metadata conditioning
    USE_ANGULAR_LOSS = True # Von Mises / Angular Cosine Loss for cyclic regression
    
    # --- Best Optuna Parameters ---
    PARAMS_CONVNEXT_TINY = {
        "dropout": 0.1555,
        "lr": 4.42e-04,
        "eta_min": 1.83e-05,
        "weight_decay": 0.0373,
        "aug_magnitude": "heavy",
        "mixup_alpha": 0.1776,
        "label_noise": 0.0437,
        "freeze_until": "features.4",
    }
    
    PARAMS_SWIN_T = {
        "dropout": 0.08979877262955549,
        "lr": 0.00023688639503640813,
        "eta_min": 5.395030966670232e-06,
        "weight_decay": 0.04123206532618727,
        "aug_magnitude": "heavy",
        "mixup_alpha": 0.0733991780504304,
        "label_noise": 0.014680559213273096,
        "freeze_until": "features.6",
    }

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
    NUM_WORKERS      = 8
    WEIGHTED_SAMPLER = True

    # --- Augmentation & Regularization Defaults -------------------------------
    DROPOUT          = 0.1
    AUG_MAGNITUDE    = "moderate"
    MIXUP_ALPHA      = 0.15
    LABEL_NOISE_STD  = 0.02

    # --- Test-Time Augmentation -----------------------------------------------
    TTA_ENABLED = True
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
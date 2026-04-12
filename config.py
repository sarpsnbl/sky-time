"""
config.py
=========
Central configuration file for Time-of-Day Estimation training and evaluation.
Adjust these parameters before running main.py or tune.py.

Changes vs previous version
----------------------------
* WARMUP_EPOCHS: linear LR warmup before cosine annealing.
  Set to 0 for CNN backbones (ConvNeXt, EfficientNet, ResNet).
  Set to 3–5 for transformer backbones (Swin, ViT).
  tune.py overrides this per-trial automatically.

* OPTUNA_MODELS: controls which backbone families are included in the
  multi-architecture HPO sweep run by tune.py.
  Each entry is a family name matching a key in tune.FAMILY_MODELS:
      "convnext"     → convnext_tiny, convnext_small, convnext_base
      "efficientnet" → efficientnet_b3, efficientnet_b4
      "swin"         → swin_t, swin_s
      "vit"          → vit_b_16
      "resnet"       → resnet50
  Run a single family:  python tune.py --family convnext
  Run all families:     python tune.py

* OPTUNA_N_STARTUP_TRIALS bumped from 5 → 8 for better random coverage
  before TPE kicks in.

* OPTUNA_EPOCHS bumped from 25 → 40 to give configs enough time to warm up
  before the pruner fires (the previous 25-epoch window was too aggressive).

* Training defaults updated to the best values found in the HPO study:
    LR=0.00406, WEIGHT_DECAY=1.58e-5, HIDDEN_DIM=320, DROPOUT=0.296,
    ETA_MIN=5.2e-6, MIXUP_ALPHA=0.046, LABEL_NOISE_STD=0.043

* MODEL now also accepts "efficientnet_b4", "swin_s".
"""


class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR  = "dataset_512"  # Flat folder containing images with EXIF data
    IMAGE_SIZE = 448            # Input image size for the model

    # --- Cross-Validation -----------------------------------------------------
    FOLD            = 0     # Which CV fold to use (0-indexed); ignored when TRAIN_ALL_FOLDS=True
    N_SPLITS        = 5     # KFold splits (set to 1 for a single random split)
    VAL_RATIO       = 0.2   # Validation fraction when N_SPLITS = 1
    TRAIN_ALL_FOLDS = True  # If True, train all N_SPLITS folds and ensemble at inference

    # --- Image Heuristic Features ---------------------------------------------
    USE_IMAGE_FEATURES = True   # Append handcrafted photometric features to metadata vector

    # --- Model ----------------------------------------------------------------
    # Supported backbones:
    #   "resnet50", "efficientnet_b3", "efficientnet_b4",
    #   "convnext_tiny", "convnext_small", "convnext_base",
    #   "swin_t", "swin_s", "vit_b_16"
    MODEL        = "convnext_tiny"
    PRETRAINED   = True
    FREEZE_UNTIL = "features.4"     # Freeze backbone up to this layer name (see main.py docstring)
    HIDDEN_DIM   = 384          # Hidden layer width for the fusion MLP
    DROPOUT      = 0.167786        # Dropout probability in the fusion MLP

    # --- Training -------------------------------------------------------------
    EPOCHS         = 80         # Number of training epochs
    WARMUP_EPOCHS  = 0          # Linear LR warmup epochs (0 = disabled; use 3–5 for Swin/ViT)
    UNFREEZE_EPOCH = 30         # Epoch to unfreeze full backbone (None = never)
    BATCH_SIZE     = 16
    LR             = 0.001896    # AdamW learning rate
    ETA_MIN        = 2.781919e-07    # Cosine scheduler minimum LR
    WEIGHT_DECAY   = 0.000241   # AdamW weight decay
    NUM_WORKERS    = 8         # DataLoader workers (adjust based on your CPU cores and memory)
    USE_AMP        = True       # Automatic mixed precision (CUDA only)
    WEIGHTED_SAMPLER = False    # Oversample underrepresented hours

    # --- Augmentation ---------------------------------------------------------
    AUG_MAGNITUDE   = "none"   # "none" | "light" | "medium" | "heavy"
    MIXUP_ALPHA     = 0.184888     # Mixup alpha; 0.0 = disabled
    LABEL_NOISE_STD = 0.042620     # Gaussian noise std on (sin_t, cos_t) targets; 0 = off

    # --- Test-Time Augmentation -----------------------------------------------
    TTA_ENABLED = True          # Average predictions over augmented copies at eval/inference
    TTA_FLIPS   = 4             # Number of TTA passes (1 = no TTA)

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR = "checkpoints"
    CHECKPOINT = None           # e.g. "checkpoints/best_fold0.pt"
    EVAL_ONLY  = False
    SEED       = 42

    # --- Optuna Hyperparameter Optimisation -----------------------------------
    # Which backbone families to include in the multi-architecture sweep.
    # Remove entries to skip families; reorder to run preferred families first.
    OPTUNA_MODELS = [
        "convnext",
        #"swin",
        #"efficientnet",
    ]

    OPTUNA_N_TRIALS          = 20    # Trials per family study
    OPTUNA_EPOCHS            = 50    # Training epochs per trial (longer = less pruner noise)
    OPTUNA_TIMEOUT_SECONDS   = None  # Wall-clock timeout per family (None = unlimited)
    OPTUNA_STUDY_NAME        = "tod_hpo"  # Base name (family suffix appended automatically)
    OPTUNA_N_STARTUP_TRIALS  = 8     # Random trials before TPE engages (was 5)
    OPTUNA_CV_FOLDS          = 1     # Folds per trial (1=fast, 3=reliable, 5=thorough)
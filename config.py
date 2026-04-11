"""
config.py
=========
Central configuration file for Time-of-Day Estimation training and evaluation.
Adjust these parameters before running Main.py.
"""

class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR = "dataset_512"   # Flat folder containing images with EXIF data
    IMAGE_SIZE = 448       # Input image size for the model

    # --- Cross-Validation ------------------------------------------------
    FOLD = 0                    # Which CV fold to use (0-indexed)
    N_SPLITS = 5                # KFold splits (set to 1 for a single random split)
    VAL_RATIO = 0.2             # Validation fraction when N_SPLITS = 1

    # --- Model ----------------------------------------------------------------
    MODEL = "convnext_tiny"     # Backbone architecture: "resnet50", "efficientnet_b3", or "convnext_tiny"
    PRETRAINED = True           # Use ImageNet pretrained weights for ResNet-50
    FREEZE_UNTIL = "layer1"     # Freeze backbone up to this layer name
    HIDDEN_DIM = 180            # Hidden layer width for the fusion MLP
    DROPOUT = 0.392             # Dropout probability in the fusion MLP

    # --- Training -------------------------------------------------------------
    EPOCHS = 50                 # Number of training epochs
    BATCH_SIZE = 24             # Training batch size
    LR = 0.00365               # Learning rate
    WEIGHT_DECAY = 0.000186     # Weight decay for AdamW
    NUM_WORKERS = 8             # DataLoader worker processes
    USE_AMP = True              # Enable automatic mixed precision (CUDA only)
    WEIGHTED_SAMPLER = False    # Oversample underrepresented hours

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR = "checkpoints"  # Directory to save model checkpoints
    CHECKPOINT = None           # Path to a checkpoint to resume from (e.g., "checkpoints/best.pt")
    EVAL_ONLY = False           # Set to True to only evaluate the model (requires CHECKPOINT)
    SEED = 42                   # Random seed for reproducibility

    # --- Optuna Hyperparameter Optimization -----------------------------------
    # Run `python tune.py` to start a study. These settings are ignored by main.py.
    OPTUNA_N_TRIALS = 30           # Total number of HPO trials to run
    OPTUNA_EPOCHS = 30             # Epochs per trial (shorter than full training for speed)
    OPTUNA_TIMEOUT_SECONDS = None  # Optional wall-clock budget in seconds (None = unlimited)
    OPTUNA_STUDY_NAME = "tod_hpo"  # Study name; reuse to resume a crashed/partial study
    OPTUNA_N_STARTUP_TRIALS = 5    # Trials before TPE sampler kicks in (random exploration)
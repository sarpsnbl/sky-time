"""
config.py
=========
Central configuration file for Time-of-Day Estimation training and evaluation.
Adjust these parameters before running Main.py.
"""

class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR = "dataset"   # Flat folder containing images with EXIF data

    # --- CV (Cross-Validation) ------------------------------------------------
    FOLD = 0                    # Which CV fold to use (0-indexed)
    N_SPLITS = 5                # KFold splits (set to 1 for a single random split)
    VAL_RATIO = 0.2             # Validation fraction when N_SPLITS = 1

    # --- Model ----------------------------------------------------------------
    PRETRAINED = True           # Use ImageNet pretrained weights for ResNet-50
    FREEZE_UNTIL = "layer2"     # Freeze backbone up to this layer name
    HIDDEN_DIM = 256            # Hidden layer width for the fusion MLP
    DROPOUT = 0.3               # Dropout probability in the fusion MLP

    # --- Training -------------------------------------------------------------
    EPOCHS = 40                 # Number of training epochs
    BATCH_SIZE = 32             # Training batch size
    LR = 1e-4                   # Learning rate
    WEIGHT_DECAY = 1e-4         # Weight decay for AdamW
    NUM_WORKERS = 4             # DataLoader worker processes
    USE_AMP = True              # Enable automatic mixed precision (CUDA only)
    WEIGHTED_SAMPLER = False    # Oversample underrepresented hours

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR = "checkpoints"  # Directory to save model checkpoints
    CHECKPOINT = None           # Path to a checkpoint to resume from (e.g., "checkpoints/best.pt")
    EVAL_ONLY = False           # Set to True to only evaluate the model (requires CHECKPOINT)
    SEED = 42                   # Random seed for reproducibility
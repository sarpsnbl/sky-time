"""
config.py
=========
Central configuration for Time-of-Day Estimation.
Focused on Swin-T with dynamic parameter decoupling.
"""
import os
import json
from typing import Optional, Dict, Any

class Config:
    # --- Data -----------------------------------------------------------------
    IMAGE_DIR  = "dataset_512"
    IMAGE_SIZE = 512

    # --- Cross-Validation -----------------------------------------------------
    FOLD            = 0     
    N_SPLITS        = 5
    VAL_RATIO       = 0.2
    TRAIN_ALL_FOLDS = True

    # --- Image Heuristic Features & Optical Physics ---------------------------
    USE_IMAGE_FEATURES = True
    USE_PCA_FEATURES   = True   # Compress 97 handcrafted features down to orthogonal PCs
    PCA_COMPONENTS     = 25     # Captures >90% variance while eliminating collinear noise
    FEATURE_CACHE_FILE = "dataset_512/heuristics_feature_cache_v2.pt"
    PCA_MODEL_FILE     = "dataset_512/heuristics_pca_v2.pt"
    COORDINATED_FLIP   = True   # Synchronize horizontal flip with directional metadata

    # --- Model & Architecture (Swin-T Focus) ----------------------------------
    MODEL            = "swin_t"
    PRETRAINED       = True
    FREEZE_UNTIL     = "features.6"
    HIDDEN_DIM       = 384
    USE_FILM         = True     # Feature-wise Linear Modulation for metadata conditioning
    LOSS_FUNCTION    = "circular_smooth_l1"  # 'circular_smooth_l1' (geodesic Huber) or 'angular_cosine'
    BETA_MINUTES     = 10.0     # Smooth transition threshold between quadratic and linear Huber penalty
    USE_ANGULAR_LOSS = True     # Backward-compatible alias for non-MSE regression

    # --- Training & Hardware Optimizations ------------------------------------
    EPOCHS           = 80
    UNFREEZE_EPOCH   = None         
    BATCH_SIZE       = 8
    ACCUM_STEPS      = 1      # Effective batch size = BATCH_SIZE * ACCUM_STEPS
    
    # Accelerated execution
    USE_AMP           = True   # Mixed Precision
    USE_COMPILE       = True   # torch.compile() for graph optimization (PT 2.0+)
    USE_CHANNELS_LAST = True   # NHWC memory format for Tensor Core speedup
    USE_8BIT_OPTIM    = True   # bitsandbytes 8-bit AdamW to save VRAM
    NUM_WORKERS      = 8
    WEIGHTED_SAMPLER = True

    # --- Augmentation & Regularization Defaults -------------------------------
    DROPOUT          = 0.0898
    LR               = 2.37e-04
    ETA_MIN          = 5.40e-06
    WEIGHT_DECAY     = 0.0412
    AUG_MAGNITUDE    = "heavy"
    MIXUP_ALPHA      = 0.0734
    LABEL_NOISE_STD  = 0.0147

    # --- Test-Time Augmentation -----------------------------------------------
    TTA_ENABLED = True
    TTA_FLIPS   = 2

    # --- I/O & Execution ------------------------------------------------------
    OUTPUT_DIR  = "checkpoints"
    FIGURES_DIR = "figures"
    CHECKPOINT  = None
    EVAL_ONLY   = False
    SEED        = 42
    PARAMS_FILE = "checkpoints/best_params_swin_t.json"

    # --- Optuna Hyperparameter Optimisation -----------------------------------
    OPTUNA_N_TRIALS         = 25
    OPTUNA_EPOCHS           = 60
    OPTUNA_TIMEOUT_SECONDS  = None
    OPTUNA_N_STARTUP_TRIALS = 5
    OPTUNA_CV_FOLDS         = 1

    # Active runtime parameters dictionary
    PARAMS: Dict[str, Any] = {}

    @classmethod
    def load_params_from_json(cls, json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads parameters from a best params JSON file (outputted by HPO)
        and populates Config runtime attributes.
        """
        path = json_path or cls.PARAMS_FILE
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Support nested {"params": {...}, "score": {...}} or flat dict
                params = data.get("params", data)
                cls.PARAMS = dict(params)

                # Sync key fields to Config class attributes
                if "lr" in params: cls.LR = float(params["lr"])
                if "weight_decay" in params: cls.WEIGHT_DECAY = float(params["weight_decay"])
                if "dropout" in params: cls.DROPOUT = float(params["dropout"])
                if "hidden_dim" in params: cls.HIDDEN_DIM = int(params["hidden_dim"])
                if "freeze_until" in params: cls.FREEZE_UNTIL = str(params["freeze_until"])
                if "eta_min" in params: cls.ETA_MIN = float(params["eta_min"])
                if "aug_magnitude" in params: cls.AUG_MAGNITUDE = str(params["aug_magnitude"])
                if "mixup_alpha" in params: cls.MIXUP_ALPHA = float(params["mixup_alpha"])
                if "label_noise" in params: cls.LABEL_NOISE_STD = float(params["label_noise"])
                if "beta_minutes" in params: cls.BETA_MINUTES = float(params["beta_minutes"])
                if "batch_size" in params: cls.BATCH_SIZE = int(params["batch_size"])
                if "accum_steps" in params: cls.ACCUM_STEPS = int(params["accum_steps"])
                if "model" in data: cls.MODEL = str(data["model"])

                return cls.PARAMS
            except Exception as e:
                print(f"[Config] Warning: Failed to parse params file '{path}': {e}. Using class defaults.")
        
        # Fallback dictionary if no file exists
        cls.PARAMS = {
            "lr": cls.LR,
            "weight_decay": cls.WEIGHT_DECAY,
            "hidden_dim": cls.HIDDEN_DIM,
            "dropout": cls.DROPOUT,
            "freeze_until": cls.FREEZE_UNTIL,
            "eta_min": cls.ETA_MIN,
            "aug_magnitude": cls.AUG_MAGNITUDE,
            "mixup_alpha": cls.MIXUP_ALPHA,
            "label_noise": cls.LABEL_NOISE_STD,
            "beta_minutes": cls.BETA_MINUTES,
            "batch_size": cls.BATCH_SIZE,
            "accum_steps": cls.ACCUM_STEPS,
        }
        return cls.PARAMS

    @classmethod
    def apply_args(cls, args) -> None:
        """
        Overrides Config and PARAMS attributes with any explicitly passed CLI arguments.
        """
        arg_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        
        # If a params file is given, load it first
        params_file = arg_dict.get("params_file")
        if params_file:
            cls.PARAMS_FILE = params_file
            cls.load_params_from_json(params_file)
        elif not cls.PARAMS:
            cls.load_params_from_json()

        # Map CLI arguments to Config / PARAMS fields
        cli_mappings = {
            "model": ("MODEL", str),
            "lr": ("LR", float),
            "weight_decay": ("WEIGHT_DECAY", float),
            "dropout": ("DROPOUT", float),
            "hidden_dim": ("HIDDEN_DIM", int),
            "freeze_until": ("FREEZE_UNTIL", str),
            "eta_min": ("ETA_MIN", float),
            "aug_magnitude": ("AUG_MAGNITUDE", str),
            "mixup_alpha": ("MIXUP_ALPHA", float),
            "label_noise": ("LABEL_NOISE_STD", float),
            "beta_minutes": ("BETA_MINUTES", float),
            "batch_size": ("BATCH_SIZE", int),
            "accum_steps": ("ACCUM_STEPS", int),
            "epochs": ("EPOCHS", int),
            "fold": ("FOLD", int),
            "train_all_folds": ("TRAIN_ALL_FOLDS", bool),
            "eval_only": ("EVAL_ONLY", bool),
            "checkpoint": ("CHECKPOINT", str),
            "loss_function": ("LOSS_FUNCTION", str),
            "image_size": ("IMAGE_SIZE", int),
            "output_dir": ("OUTPUT_DIR", str),
        }

        for cli_key, (attr_name, cast_fn) in cli_mappings.items():
            val = arg_dict.get(cli_key)
            if val is not None:
                cast_val = cast_fn(val)
                setattr(cls, attr_name, cast_val)
                # Keep PARAMS in sync
                param_key = cli_key if cli_key in cls.PARAMS else cli_key.lower()
                cls.PARAMS[param_key] = cast_val

        if arg_dict.get("no_compile") is True:
            cls.USE_COMPILE = False

# Auto-initialize params
Config.load_params_from_json()
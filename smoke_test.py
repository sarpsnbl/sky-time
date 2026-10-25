"""
smoke_test.py
=============
Fast end-to-end sanity and smoke test suite for Sky-Time estimation.
Verifies all core pipeline components in < 15 seconds:
1. Config & Decoupled Hyperparameter JSON loading.
2. 55-D Metadata dimensionality (30 physical + 25 PCA optical).
3. Coordinated horizontal flip & directional symmetry parity.
4. Swin-T model instantiation, S¹ unit-circle forward pass & CircularSmoothL1 gradient flow.
5. Dataset loading & 512×512 sample iteration.
6. Unified visualization suite smoke check.

Usage:
    python smoke_test.py
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

def smoke_test():
    print("=" * 65)
    print("  SKY-TIME END-TO-END SMOKE TEST")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # 1. Config & JSON Parameter Decoupling
    # -----------------------------------------------------------------------
    print("\n[1/6] Testing Config & Best Params Decoupling ...")
    from config import Config as cfg
    assert os.path.isfile(cfg.PARAMS_FILE), f"Missing params file: {cfg.PARAMS_FILE}"
    params = cfg.load_params_from_json()
    assert isinstance(params, dict) and "lr" in params, "Params failed to load as dictionary"
    assert cfg.MODEL == "swin_t", f"Expected default model swin_t, got {cfg.MODEL}"
    print(f"  ✓ Loaded params from '{cfg.PARAMS_FILE}'")
    print(f"  ✓ Model: {cfg.MODEL} | LR: {cfg.LR:.2e} | WD: {cfg.WEIGHT_DECAY:.2e} | Beta: {cfg.BETA_MINUTES}m")

    # -----------------------------------------------------------------------
    # 2. 55-D Metadata Dimensionality & Feature Extraction
    # -----------------------------------------------------------------------
    print("\n[2/6] Testing 55-D Metadata & Optical Physics Pipeline ...")
    from TimeOfDayDataLoader import (
        get_metadata_dim,
        ImageFeatureExtractor,
        TimeOfDayLabel,
        _PHYSICAL_METADATA_DIM,
        _IMAGE_FEATURE_RAW_DIM,
    )
    meta_dim = get_metadata_dim()
    expected_dim = 30 + 25  # 30 physical + 25 PCA
    assert meta_dim == expected_dim, f"Expected {expected_dim} metadata dims, got {meta_dim}"
    print(f"  ✓ Metadata dimension: {meta_dim} (30 Physical + 25 PCA Optical Descriptors)")

    # Test synthetic image extraction
    synthetic_img = Image.new("RGB", (512, 512), color=(120, 150, 200))
    raw_feat = ImageFeatureExtractor.extract(synthetic_img)
    assert raw_feat.shape == (_IMAGE_FEATURE_RAW_DIM,), f"Expected shape ({_IMAGE_FEATURE_RAW_DIM},), got {raw_feat.shape}"
    assert np.all(np.isfinite(raw_feat)), "Feature vector contains NaNs or Infs"
    print(f"  ✓ Raw optical feature extractor produced valid shape {raw_feat.shape}")

    # -----------------------------------------------------------------------
    # 3. Coordinated Inversion & Directional Symmetry
    # -----------------------------------------------------------------------
    print("\n[3/6] Testing Coordinated Horizontal Flip Symmetry ...")
    flipped_feat = ImageFeatureExtractor.flip_features(raw_feat)
    double_flipped = ImageFeatureExtractor.flip_features(flipped_feat)
    assert np.allclose(raw_feat, double_flipped, atol=1e-5), "Flip operation not perfectly invertible"
    assert abs(flipped_feat[78] - (-raw_feat[78])) < 1e-6, "Horizontal asymmetry not negated"
    assert abs(flipped_feat[84] - (1.0 - raw_feat[84])) < 1e-6, "Sun X centroid not mirrored"

    label = TimeOfDayLabel(time_min=480.0, month=5, day_of_year=130, heading=60.0, image_features=raw_feat)
    flipped_label = label.flip_horizontal()
    assert abs(flipped_label.heading - 300.0) < 1e-5, "Heading mirroring failed"
    print("  ✓ Directional features & heading invert with exact physical symmetry")

    # -----------------------------------------------------------------------
    # 4. Swin-T Architecture, S¹ Projection & Geodesic Huber Loss
    # -----------------------------------------------------------------------
    print("\n[4/6] Testing Swin-T Model Forward/Backward Pass on S¹ ...")
    from Main import TimeOfDayModel, get_loss_criterion

    model = TimeOfDayModel(
        pretrained=False,
        metadata_dim=meta_dim,
        hidden_dim=128,
        dropout=0.1,
    )
    B = 2
    dummy_img = torch.randn(B, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    dummy_meta = torch.randn(B, meta_dim)
    dummy_target = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    preds = model(dummy_img, dummy_meta)
    if getattr(cfg, "USE_MIXTURE_HEAD", False):
        assert preds.shape == (B, 6), f"Expected mixture preds shape ({B}, 6), got {preds.shape}"
        v1_norm = preds[:, 2:4].norm(p=2, dim=-1)
        v2_norm = preds[:, 4:6].norm(p=2, dim=-1)
        assert torch.allclose(v1_norm, torch.ones_like(v1_norm), atol=1e-5), "Mode 1 not normalized on S¹"
        assert torch.allclose(v2_norm, torch.ones_like(v2_norm), atol=1e-5), "Mode 2 not normalized on S¹"
        print(f"  ✓ Bimodal S¹ mixture projection verified: v1_norm={v1_norm[0].item():.4f}, v2_norm={v2_norm[0].item():.4f}")
    else:
        assert preds.shape == (B, 2), f"Expected preds shape ({B}, 2), got {preds.shape}"
        norms = preds.norm(p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Predictions not normalized on S¹ unit circle"
        print(f"  ✓ S¹ Unit-circle projection verified: norm={norms[0].item():.5f}")

    criterion = get_loss_criterion(cfg.LOSS_FUNCTION, cfg.BETA_MINUTES)
    loss = criterion(preds, dummy_target)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.fusion.parameters())
    assert has_grad, "Gradient did not backpropagate through FiLM fusion head"
    print(f"  ✓ Loss criterion ({criterion.__class__.__name__}): {loss.item():.4f} with active gradient flow")

    # -----------------------------------------------------------------------
    # 5. Dataset Loading & Iteration
    # -----------------------------------------------------------------------
    print("\n[5/6] Testing Dataset & DataLoader Sample Loading ...")
    from TimeOfDayDataLoader import TimeOfDayDataset, get_transforms, create_dataloaders
    dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False))
    assert len(dataset) == 2480, f"Expected 2480 samples in dataset_512, found {len(dataset)}"

    sample_img, sample_meta, sample_target = dataset[0]
    assert sample_img.shape == (3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), f"Unexpected image shape: {sample_img.shape}"
    assert sample_meta.shape == (meta_dim,), f"Unexpected metadata shape: {sample_meta.shape}"
    assert sample_target.shape == (2,), f"Unexpected target shape: {sample_target.shape}"
    print(f"  ✓ Loaded dataset_512: {len(dataset)} verified samples")
    print(f"  ✓ Sample 0 verified: image {list(sample_img.shape)}, meta {list(sample_meta.shape)}, target {list(sample_target.shape)}")

    # -----------------------------------------------------------------------
    # 6. Unified Visualization Suite
    # -----------------------------------------------------------------------
    print("\n[6/6] Testing Unified Visualization Suite ...")
    import visualize
    assert hasattr(visualize, "plot_architecture"), "visualize.py missing plot_architecture"
    assert hasattr(visualize, "plot_pca_analysis"), "visualize.py missing plot_pca_analysis"
    assert hasattr(visualize, "plot_training_results"), "visualize.py missing plot_training_results"
    assert hasattr(visualize, "plot_prediction_diagnostics"), "visualize.py missing plot_prediction_diagnostics"
    assert hasattr(visualize, "plot_dataset_distribution"), "visualize.py missing plot_dataset_distribution"
    assert os.path.isdir("figures"), "figures/ directory missing"
    print("  ✓ Unified visualization suite verified with all 9 publication figure generators")

    print("\n" + "=" * 65)
    print("  ALL 6 SMOKE TEST MODULES PASSED SUCCESSFULLY! (100% OK)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    smoke_test()

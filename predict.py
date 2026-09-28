"""
predict.py
==========
High-precision single image and batch inference script for:
    Deep Learning-Based Time-of-Day Estimation
    from Sky Images and Comprehensive EXIF Metadata

Usage:
    python predict.py
    python predict.py --image path/to/sky.jpg
    python predict.py --image path/to/sky.jpg --checkpoint checkpoints/best_swin_t_fold0.pt
"""
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image

from config import Config as cfg
from TimeOfDayDataLoader import (
    extract_exif_data,
    TimeOfDayLabel,
    ImageFeatureExtractor,
    decode_time_tensor,
    minutes_to_hhmm,
    tta_predict,
    compute_astronomy,
    MINUTES_PER_DAY,
)
from Main import TimeOfDayModel, load_checkpoint


def get_inference_transform(image_size: int = 512):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_single_image(
    image_path: str,
    model: TimeOfDayModel,
    device: torch.device,
    extractor: ImageFeatureExtractor,
    transform: transforms.Compose,
    use_tta: bool = True,
    n_passes: int = 2,
):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Extract complete multi-tier EXIF metadata
    exif = extract_exif_data(image_path)
    if exif is None:
        print(f"Warning: Could not read EXIF data for {image_path}. Using fallback values.")
        # Minimal fallback
        exif = extract_exif_data._dummy() if hasattr(extract_exif_data, "_dummy") else None

    # Load and transform image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Extract 80 handcrafted atmospheric and photometric features
    heuristics = extractor.extract(img)

    # Compute astronomical parameters
    day_of_year = 172
    if exif is not None and exif.month is not None and exif.day is not None:
        from TimeOfDayDataLoader import _day_of_year
        day_of_year = _day_of_year(exif.month, exif.day, exif.year or 2024)

    month = exif.month if (exif and exif.month) else 6
    lat = exif.lat if exif else None
    lon = exif.lon if exif else None
    tz  = exif.tz_offset if exif else 3.0
    solar_dec, eot, day_len, solar_noon, max_elev = compute_astronomy(month, day_of_year, lat, lon, tz)

    # Create label container to assemble 108-dim tensor
    label = TimeOfDayLabel(
        time_min=exif.time_min if exif else 720.0,
        month=exif.month if exif else 6,
        day_of_year=day_of_year,
        latitude=exif.lat if exif else None,
        longitude=exif.lon if exif else None,
        day_of_week=exif.day_of_week if exif else None,
        tz_offset=exif.tz_offset if exif else 3.0,
        heading=exif.heading if exif else None,
        altitude=exif.altitude if exif else None,
        ev100=exif.ev100 if exif else None,
        brightness_val=exif.brightness_val if exif else None,
        shutter_speed=exif.shutter_speed if exif else None,
        iso=exif.iso if exif else None,
        f_number=exif.f_number if exif else None,
        exposure_bias=exif.exposure_bias if exif else 0.0,
        fov=exif.fov if exif else None,
        image_features=heuristics,
    )
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        if use_tta:
            preds = tta_predict(model, img_tensor, meta_tensor, n_passes=n_passes)
        else:
            preds = model(img_tensor, meta_tensor)

        pred_min = decode_time_tensor(preds.cpu()).item()

    print(f"\n=======================================================")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"=======================================================")
    if exif and exif.time_min is not None:
        diff = abs(pred_min - exif.time_min)
        circ_err = min(diff, MINUTES_PER_DAY - diff)
        print(f"  Ground-Truth EXIF Time:   {minutes_to_hhmm(exif.time_min)} ({exif.time_min:.1f} min)")
        print(f"  Predicted Time of Day:    {minutes_to_hhmm(pred_min)} ({pred_min:.1f} min)")
        print(f"  Absolute Angular Error:   {circ_err:.1f} minutes")
    else:
        print(f"  Predicted Time of Day:    {minutes_to_hhmm(pred_min)} ({pred_min:.1f} min)")

    print(f"-------------------------------------------------------")
    print(f"  [Astronomical Invariants]")
    print(f"    Solar Declination (delta): {solar_dec:+.2f} deg")
    print(f"    Equation of Time (EoT):   {eot:+.2f} min")
    print(f"    Local Solar Noon:         {minutes_to_hhmm(solar_noon)}")
    print(f"    Day Length:               {day_len:.2f} hours")
    print(f"    Max Elevation (alpha_max):{max_elev:.2f} deg")

    print(f"  [Camera & Scene Photometrics]")
    ev_str = f"{exif.ev100:.2f}" if (exif and exif.ev100 is not None) else "N/A"
    bv_str = f"{exif.brightness_val:.2f}" if (exif and exif.brightness_val is not None) else "N/A"
    head_str = f"{exif.heading:.1f} deg" if (exif and exif.heading is not None) else "N/A"
    print(f"    Exposure Value (EV100):   {ev_str}")
    print(f"    APEX Brightness (Bv):     {bv_str}")
    print(f"    Compass Heading:          {head_str}")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Estimate time of day from sky image and EXIF.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")

    # Build model with FiLM and 108-dim metadata
    model = TimeOfDayModel(
        pretrained=False,
        hidden_dim=cfg.HIDDEN_DIM,
        metadata_dim=108,
        use_film=cfg.USE_FILM,
    ).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
    else:
        print("Note: Running with initialized model weights (provide --checkpoint for trained weights).")

    model.eval()
    extractor = ImageFeatureExtractor()
    transform = get_inference_transform(cfg.IMAGE_SIZE)

    if args.image:
        predict_single_image(args.image, model, device, extractor, transform, use_tta=not args.no_tta)
    else:
        # Default test on predict1.jpg and predict2.jpg if present
        for test_img in ["predict1.jpg", "predict2.jpg"]:
            if os.path.exists(test_img):
                predict_single_image(test_img, model, device, extractor, transform, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()

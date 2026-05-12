# Deep Learning-Based Time-of-Day Estimation from Sky Images and EXIF Calendar Dates

## Abstract
Automated image organization systems in modern digital galleries frequently rely on simplistic brightness-based heuristics to classify outdoor photographs by time of day. This approach leads to systematic misclassifications under artificial lighting and overcast conditions. This project presents a hybrid deep learning framework that estimates the exact time of capture from a single sky image by fusing visual features with cyclically encoded EXIF calendar metadata.

## Methodology
The framework evaluates two primary backbone architectures in a Python/PyTorch pipeline: ConvNeXt-Tiny and Swin Transformer (Swin-T). These are paired with a metadata-fusion multilayer perceptron (MLP) operating on 77 handcrafted photometric descriptors (including color-space statistics, sky-region luminance gradients, and edge density measures).

To respect the circular topology of the 24-hour clock and the annual calendar, a cyclic sine-cosine target encoding is employed. This eliminates discontinuities at the midnight and year boundaries that plague linear time representations.

For comparative analysis, a parallel MATLAB pipeline utilizing SqueezeNet and a purely statistical ensemble combining Support Vector Regression (SVR), Gradient Boosted Trees (GBT), and Random Forest (RF) on handcrafted features serve as baselines.

## Dataset
The dataset comprises 2,483 consumer smartphone photographs collected under diverse conditions spanning dawn, midday, dusk, and artificial-light scenes. Each image contains visible sky content and valid EXIF metadata, including capture time and calendar date.

## Results
In a five-fold cross-validation evaluation, the Swin-T architecture achieved the best performance with a mean cyclic Mean Absolute Error (MAE) of 52.45 minutes, outperforming the ConvNeXt-Tiny model and the baseline approaches. The results demonstrate the efficacy of fusing learned visual representations with temporal metadata for robust continuous time-of-day regression.

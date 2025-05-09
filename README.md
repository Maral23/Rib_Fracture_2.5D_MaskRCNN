# 2.5D Mask R-CNN for Rib Fracture Detection

This repository implements a 2.5D Mask R-CNN model for detecting and classifying rib fractures in CT scan slices. By processing triplets of consecutive slices (z−1, z, z+1), the model leverages contextual information to improve detection accuracy while maintaining computational efficiency.

> **Note:** All components—preprocessing, model definition, training, and evaluation—are implemented in a single Jupyter notebook located at `2.5D MaskRCNN.ipynb`.

---

## Project Structure

* **`data/raw/`**: Raw CT scan and label data.
* **`data/sample of preprocessed data/`**: Preprocessed tensor data (.pt files).
* * **`2.5D MaskRCNN with less visualization.ipynb`**: Main notebook with all code and experiments. 
* **`notebooks/2.5D MaskRCNN.ipynb`**: Same notebook but with 300 visualized slices. Takes a while to open since its a big file.
* **`requirements.txt`**: List of Python dependencies.

---

## Libraries & Setup

The project uses the following libraries:

* `torch`, `torchvision` – Deep learning with Mask R-CNN
* `nibabel` – Loading NIfTI-format CT scans
* `scikit-image` – Image resizing
* `pandas` – Label mapping from CSV
* `torchmetrics` – Evaluation metrics

Google Drive integration is included for seamless dataset access when using Google Colab.

---

## Preprocessing

CT volumes are first windowed using Hounsfield Unit (HU) windowing (`level=450`, `width=1300`) to enhance bone contrast. Intensities are then normalized to the \[0, 1] range.

Fracture labels are remapped using `ribfrac-train-info-1.csv`. Each sample consists of three consecutive slices centered at the z-index, resized to 256×256 pixels. Tiny fracture masks (<5 pixels) are filtered out, and a dummy 8×8 bounding box is added when no fracture is present to stabilize training.

All outputs are saved as `.pt` tensors for efficient loading.

---

## Data Loading & Augmentation

A custom `PreprocessedRibFractureDataset` class handles loading and parsing of the `.pt` tensors. It uses `WeightedRandomSampler` to handle class imbalance, and applies integrity checks on tensor shapes, bounding boxes, and label values.

Data augmentation includes synchronized random horizontal flips for images, masks, and bounding boxes. DataLoaders are configured with:

* **Batch Size**: 8 (train), 4 (test)
* **Workers**: 4 subprocesses for parallel loading

---

## Model Architecture

The model builds on Mask R-CNN with several customizations:

* **EnhancedSliceAwareConv**: Replaces the first convolutional layer to process each 2D slice independently before fusing features using cross-slice attention and residual connections.
* **Custom Anchors**: Region Proposal Network (RPN) uses anchors optimized for rib fractures: `[4, 8, 12, 18, 32]`.
* **Improved Mask Head**: A deeper mask head with additional `BatchNorm` and `ReLU` layers.
* **Class-Biased Initialization**: Tailored weight initialization for fracture-specific features.
* **Partial Freezing**: The early backbone layer (`backbone.body.layer1`) is frozen to retain low-level spatial features.

---

## Training

Training is performed using the AdamW optimizer:

* **Initial Learning Rate**: `1e-5`, with a 10-epoch linear warmup to `3e-4`
* **Scheduler**: Cosine annealing after warmup
* **Gradient Accumulation**: Over 4 steps to simulate large batch sizes
* **Loss Weighting**: Emphasis on:

  * Mask loss ×1.5
  * RPN loss ×3.0

Model checkpoints are saved based on a composite validation metric combining FROC AUC, mAP, and Dice score.

---

## Evaluation Metrics

Model performance is measured with:

* **mAP**: Mean Average Precision at IoU thresholds of 0.3 and 0.5
* **FROC AUC**: Sensitivity vs. number of false positives per image
* **Dice Score**: For evaluating segmentation mask overlap

Class-wise performance is also reported for the following fracture types:

1. Displaced
2. Non-displaced
3. Buckle
4. Segmental
5. Unidentified

---

## Results & Interpretation

The best model—selected using the composite metric—achieved:

* \~2–3 false positives per image
* **Highest Dice (≈0.70)** for Class 2 (Non-displaced)
* **Lowest Dice (≈0.30)** for Class 5 (Unidentified)

---

## Usage

Open and run the notebook `notebooks/2.5D MaskRCNN.ipynb` sequentially to execute the pipeline. Key function calls:

```python
# Preprocessing
preprocess_and_save_tensor(raw_ct_dir, raw_label_dir, output_dir)

# Model creation & training
model = create_ribfrac_model(num_classes=6, device='cuda')
train_model(model, train_loader, val_loader, device='cuda', num_epochs=20)

# Evaluation
metrics = evaluate(model, test_loader, device='cuda')
print_metrics(metrics, title='Test Metrics')
```

---

## Future Improvements

* **3D Convolutions**: Use full volumetric context for improved spatial reasoning
* **Larger Dataset**: Expand fracture diversity across patient demographics
* **Post-Processing**: Use connected-components analysis to remove tiny false positives

---

## Dependencies

See `requirements.txt` for the complete list of packages.

> **Note:** A CUDA-enabled GPU is required for training the model efficiently.

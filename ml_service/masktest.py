# masktest.py
import os
import multiprocessing

# 1. Prevent forkserver/spawn recursion issues on Linux
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

import tempfile
import numpy as np
import tifffile
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.transform import resize
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def save_nii(arr, path):
    """Saves a 3D numpy array in nnU-Net expected orientation (Y, X, Z)."""
    transposed = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
    nib.save(nib.Nifti1Image(transposed, affine), path)


def keep_largest_component(binary_mask):
    """Post-processing: retains only the largest 3D connected component."""
    labeled, num_features = label(binary_mask > 0)
    if num_features <= 1:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    biggest_blob = sizes.argmax()
    return labeled == biggest_blob


def calculate_dice(gt, pred):
    inter = 2.0 * np.sum(gt & pred)
    total = np.sum(gt) + np.sum(pred)
    return (inter / total * 100.0) if total > 0 else 100.0


def main():
    # =========================================================================
    # 1. CONFIGURATION & FILE PATHS
    # =========================================================================
    T1_PATH = "./test_data/patient_T1.tif"
    T2_PATH = "./test_data/patient_T2.tif"
    MASK_PATH = "./test_data/patient_T1_MASK.tif"
    MODEL_DIR = "ml_service/ml_service/models/nnunet"
    OUTPUT_DIR = "./test_data"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(T1_PATH) or not os.path.exists(MASK_PATH):
        raise FileNotFoundError(f"Ensure both {T1_PATH} and {MASK_PATH} exist.")

    # =========================================================================
    # 2. LOAD SCANS & ALIGN SHAPES
    # =========================================================================
    print("Loading image data...")
    t1 = tifffile.imread(T1_PATH).astype(np.float32)
    gt_mask = tifffile.imread(MASK_PATH) > 0
    orig_shape = t1.shape

    if os.path.exists(T2_PATH):
        t2 = tifffile.imread(T2_PATH).astype(np.float32)
        if t2.shape != orig_shape:
            print(f"Resizing T2 from {t2.shape} to match T1 {orig_shape}...")
            t2 = resize(t2, orig_shape, order=3, preserve_range=True, anti_aliasing=True)
        print("[✓] Using real, aligned T2 scan.")
    else:
        print("[!] T2 scan not found. Duplicating T1 for channel 1 fallback.")
        t2 = t1.copy()

    # =========================================================================
    # 3. COMPUTE ROI BOUNDING BOX (Centered on True Lesion)
    # =========================================================================
    coords = np.argwhere(gt_mask)
    if len(coords) == 0:
        raise ValueError("Ground-truth mask has 0 foreground pixels.")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    padding = 48
    bbox = (
        slice(max(0, z_min - 1), min(orig_shape[0], z_max + 2)),
        slice(max(0, y_min - padding), min(orig_shape[1], y_max + padding)),
        slice(max(0, x_min - padding), min(orig_shape[2], x_max + padding)),
    )
    print(f"[✓] ROI bounding box: {bbox}")

    crop_t1 = t1[bbox]
    crop_t2 = t2[bbox]

    # =========================================================================
    # 4. RUN nnU-Net PREDICTION
    # =========================================================================
    print(f"Initializing nnU-Net predictor from: {MODEL_DIR}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_DIR, 
        use_folds=(0,), 
        checkpoint_name="checkpoint_best.pth"
    )

    with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
        save_nii(crop_t1, os.path.join(in_dir, "case_0000.nii.gz"))
        save_nii(crop_t2, os.path.join(in_dir, "case_0001.nii.gz"))

        print("Executing inference...")
        predictor.predict_from_files(
            in_dir,
            out_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )

        pred_crop_nii = nib.load(os.path.join(out_dir, "case.nii.gz")).get_fdata()
        # Convert back from (Y, X, Z) to (Z, Y, X)
        pred_crop = np.transpose(pred_crop_nii, (2, 0, 1)) > 0

    # =========================================================================
    # 5. UN-CROP & POST-PROCESSING
    # =========================================================================
    # Paste raw prediction into full-size array
    full_pred_raw = np.zeros_like(gt_mask, dtype=bool)
    full_pred_raw[bbox] = pred_crop

    # Keep largest connected component to clear isolated false-positive noise
    full_pred_cleaned = keep_largest_component(full_pred_raw)

    # =========================================================================
    # 6. EVALUATION METRICS
    # =========================================================================
    raw_dice = calculate_dice(gt_mask, full_pred_raw)
    cleaned_dice = calculate_dice(gt_mask, full_pred_cleaned)

    print("\n" + "=" * 65)
    print("SLICE-BY-SLICE COMPARISON")
    print("=" * 65)
    print(f"{'Slice':<8} | {'Ground Truth':<15} | {'Prediction':<15} | {'Match'}")
    print("-" * 65)

    for i in range(orig_shape[0]):
        gt_cnt = int(np.sum(gt_mask[i]))
        pred_cnt = int(np.sum(full_pred_cleaned[i]))
        status = "✓" if (gt_cnt > 0 and pred_cnt > 0) or (gt_cnt == 0 and pred_cnt == 0) else "MISMATCH"
        print(f"Slice #{i:<3} | {gt_cnt:<15} | {pred_cnt:<15} | {status}")

    print("-" * 65)
    print(f"Raw nnU-Net 3D Dice Score:     {raw_dice:.2f}%")
    print(f"Cleaned (Largest Component):   {cleaned_dice:.2f}%")
    print("=" * 65)

    # =========================================================================
    # 7. SAVE FINAL OUTPUTS & VISUALIZATION
    # =========================================================================
    # Save multi-page TIFF
    out_tif = os.path.join(OUTPUT_DIR, "predicted_mask.tif")
    tifffile.imwrite(out_tif, (full_pred_cleaned * 255).astype(np.uint8), compression="zlib")
    print(f"[✓] Saved TIFF mask: {out_tif}")

    # Save NIfTI
    out_nii = os.path.join(OUTPUT_DIR, "predicted_mask.nii.gz")
    save_nii(full_pred_cleaned.astype(np.uint8), out_nii)
    print(f"[✓] Saved NIfTI mask: {out_nii}")

    # Generate preview comparison image
    tumor_counts = np.sum(gt_mask, axis=(1, 2))
    best_slice_idx = int(np.argmax(tumor_counts)) if np.max(tumor_counts) > 0 else orig_shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")

    # Panel 1: Raw T1 MRI
    axes[0].imshow(t1[best_slice_idx], cmap="gray")
    axes[0].set_title(f"T1 MRI (Slice #{best_slice_idx})", color="white", fontsize=14)
    axes[0].axis("off")

    # Panel 2: Ground Truth Overlay
    axes[1].imshow(t1[best_slice_idx], cmap="gray")
    gt_overlay = np.ma.masked_where(~gt_mask[best_slice_idx], gt_mask[best_slice_idx])
    axes[1].imshow(gt_overlay, cmap="winter", alpha=0.6)
    axes[1].set_title("Ground Truth", color="white", fontsize=14)
    axes[1].axis("off")

    # Panel 3: Prediction Overlay
    axes[2].imshow(t1[best_slice_idx], cmap="gray")
    pred_overlay = np.ma.masked_where(~full_pred_cleaned[best_slice_idx], full_pred_cleaned[best_slice_idx])
    axes[2].imshow(pred_overlay, cmap="autumn", alpha=0.6)
    axes[2].set_title(f"nnU-Net Prediction (Dice: {cleaned_dice:.1f}%)", color="white", fontsize=14)
    axes[2].axis("off")

    preview_path = os.path.join(OUTPUT_DIR, "prediction_preview.png")
    plt.tight_layout()
    plt.savefig(preview_path, facecolor="black", dpi=150)
    plt.close()
    print(f"[✓] Saved visual preview image: {preview_path}\n")


if __name__ == "__main__":
    main()
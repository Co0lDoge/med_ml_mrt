import os
import sys
import tempfile
import numpy as np
import tifffile
import nibabel as nib
import torch
from scipy.ndimage import label
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

MODEL_DIR = "/mnt/d/Study/Projects/med_ml_mrt/ml_service/ml_service/models/nnunet"
T1_PATH = "/mnt/d/Study/Projects/med_ml_mrt/_test_data/patient_T1.tif"
MASK_PATH = "/mnt/d/Study/Projects/med_ml_mrt/_test_data/patient_T1_MASK.tif"

def save_nii(arr, path):
    transposed = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
    nib.save(nib.Nifti1Image(transposed, affine), path)

def keep_largest_component(binary_mask):
    labeled, num_features = label(binary_mask > 0)
    if num_features <= 1:
        return binary_mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    biggest_blob = sizes.argmax()
    return labeled == biggest_blob

def calculate_dice(gt, pred):
    inter = 2.0 * np.sum(gt & pred)
    total = np.sum(gt) + np.sum(pred)
    return (inter / total * 100.0) if total > 0 else 100.0

def main():
    print("=" * 70)
    print("EVALUATING MODEL PREDICTION ON DIFFERENT CHANNEL CONFIGURATIONS")
    print("=" * 70)

    t1 = tifffile.imread(T1_PATH).astype(np.float32)
    gt_mask = tifffile.imread(MASK_PATH) > 0
    orig_shape = t1.shape
    print(f"Loaded T1 scan: shape={orig_shape}, min={t1.min()}, max={t1.max()}, mean={t1.mean():.1f}")
    print(f"Ground Truth mask: total positive voxels={np.sum(gt_mask)}")

    # ROI normalized crop from nnunet_segmentation.py
    roi_normalized = (0.352, 0.666, 0.346, 0.686)
    num_slices, h, w = orig_shape
    bbox = (
        slice(0, num_slices),
        slice(int(roi_normalized[0] * h), int(roi_normalized[1] * h)),
        slice(int(roi_normalized[2] * w), int(roi_normalized[3] * w))
    )
    print(f"ROI Bounding Box: {bbox}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        MODEL_DIR, 
        use_folds=(0,), 
        checkpoint_name="checkpoint_best.pth"
    )

    experiments = [
        ("T1 in Channel 0 (T1c) + T1 in Channel 1 (T2) [Current Pipeline]", t1, t1),
        ("T1 in Channel 0 (T1c) + Zeros in Channel 1 (T2)", t1, np.zeros_like(t1)),
        ("Zeros in Channel 0 (T1c) + T1 in Channel 1 (T2) [T1 treated as T2 only]", np.zeros_like(t1), t1),
        ("T1 in Channel 0 + Inverted/Simulated T2 in Channel 1", t1, t1.max() - t1),
    ]

    results = []

    for name, vol_ch0, vol_ch1 in experiments:
        crop_ch0 = vol_ch0[bbox]
        crop_ch1 = vol_ch1[bbox]

        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            save_nii(crop_ch0, os.path.join(in_dir, "case_0000.nii.gz"))
            save_nii(crop_ch1, os.path.join(in_dir, "case_0001.nii.gz"))

            predictor.predict_from_files(
                in_dir,
                out_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )

            pred_crop_nii = nib.load(os.path.join(out_dir, "case.nii.gz")).get_fdata()
            pred_crop = np.transpose(pred_crop_nii, (2, 0, 1)) > 0

        full_pred = np.zeros_like(gt_mask, dtype=bool)
        full_pred[bbox] = pred_crop
        full_cleaned = keep_largest_component(full_pred)

        raw_dice = calculate_dice(gt_mask, full_pred)
        cleaned_dice = calculate_dice(gt_mask, full_cleaned)
        pred_voxels = np.sum(full_cleaned)

        results.append({
            "name": name,
            "raw_dice": raw_dice,
            "cleaned_dice": cleaned_dice,
            "pred_voxels": pred_voxels
        })

    print("\n" + "=" * 90)
    print(f"{'Experiment':<55} | {'Raw Dice':<10} | {'Clean Dice':<10} | {'Pred Voxels'}")
    print("=" * 90)
    for r in results:
        print(f"{r['name']:<55} | {r['raw_dice']:>8.2f}% | {r['cleaned_dice']:>8.2f}% | {r['pred_voxels']:>10}")
    print("=" * 90)

if __name__ == "__main__":
    main()


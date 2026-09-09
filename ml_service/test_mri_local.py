# test_mri_local.py
import os
import multiprocessing

# Prevent forkserver / spawn recursion issues
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

# Mock S3 secrets for Pydantic before importing any service modules
os.environ.setdefault("S3_SECRET_KEY", "dummy_secret_key")
os.environ.setdefault("S3_ACCESS_KEY", "dummy_access_key")

# Point to local nnU-Net artifacts
NNUNET_RESULTS_DIR = "ml_service/ml_service/models/nnunet"
os.environ["nnUNet_results"] = NNUNET_RESULTS_DIR
os.environ["nnUNet_raw"] = "/tmp/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/tmp/nnUNet_preprocessed"

import numpy as np
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
from ml_service.internal.ml_model.nnunet_segmentation import NnUNetSegmentationModel
import ml_service.internal.usecases.mri.mri as usecasemri


# ==============================================================================
# VISUALIZATION & EVALUATION HELPER
# ==============================================================================
def save_and_evaluate(parsed_slices, result_masks, output_dir="./test_data", gt_mask_path=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Assemble full-volume 3D binary mask (Z, Y, X)
    mri_vol = np.stack(parsed_slices, axis=0)
    pred_vol = np.stack(result_masks, axis=0) > 0
    num_slices = mri_vol.shape[0]

    # 2. Save as multi-page TIFF
    tif_path = os.path.join(output_dir, "predicted_mask.tif")
    tifffile.imwrite(tif_path, (pred_vol * 255).astype(np.uint8), compression="zlib")
    print(f"[✓] Saved Multi-page TIFF: {tif_path}")

    # 3. Save as NIfTI (.nii.gz)
    affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
    mask_nii = nib.Nifti1Image(np.transpose(pred_vol.astype(np.uint8), (1, 2, 0)), affine)
    nii_path = os.path.join(output_dir, "predicted_mask.nii.gz")
    nib.save(mask_nii, nii_path)
    print(f"[✓] Saved 3D NIfTI:        {nii_path}")

    # 4. Optional Ground Truth Comparison
    gt_vol = None
    dice_score = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_vol = tifffile.imread(gt_mask_path) > 0
        if gt_vol.shape == pred_vol.shape:
            intersection = 2.0 * np.sum(gt_vol & pred_vol)
            total = np.sum(gt_vol) + np.sum(pred_vol)
            dice_score = (intersection / total * 100.0) if total > 0 else 100.0

            print("\n" + "=" * 60)
            print("SLICE-BY-SLICE VALIDATION AUDIT")
            print("=" * 60)
            print(f"{'Slice':<8} | {'Ground Truth':<15} | {'Prediction':<15} | {'Match'}")
            print("-" * 60)
            for i in range(num_slices):
                gt_cnt = int(np.sum(gt_vol[i]))
                pred_cnt = int(np.sum(pred_vol[i]))
                status = "✓" if (gt_cnt > 0 and pred_cnt > 0) or (gt_cnt == 0 and pred_cnt == 0) else "MISMATCH"
                print(f"Slice #{i:<3} | {gt_cnt:<15} | {pred_cnt:<15} | {status}")
            print("-" * 60)
            print(f"Overall 3D Dice Score: {dice_score:.2f}%")
            print("=" * 60)

    # 5. Generate Visual Preview Panel
    if gt_vol is not None:
        best_slice_idx = int(np.argmax(np.sum(gt_vol, axis=(1, 2))))
    else:
        best_slice_idx = int(np.argmax(np.sum(pred_vol, axis=(1, 2)))) if np.sum(pred_vol) > 0 else num_slices // 2

    fig, axes = plt.subplots(1, 3 if gt_vol is not None else 2, figsize=(18 if gt_vol is not None else 12, 6), facecolor="black")

    # Panel 1: MRI
    axes[0].imshow(mri_vol[best_slice_idx], cmap="gray")
    axes[0].set_title(f"T1 MRI (Slice #{best_slice_idx})", color="white", fontsize=14)
    axes[0].axis("off")

    if gt_vol is not None:
        # Panel 2: GT Overlay
        axes[1].imshow(mri_vol[best_slice_idx], cmap="gray")
        gt_overlay = np.ma.masked_where(~gt_vol[best_slice_idx], gt_vol[best_slice_idx])
        axes[1].imshow(gt_overlay, cmap="winter", alpha=0.6)
        axes[1].set_title("Ground Truth", color="white", fontsize=14)
        axes[1].axis("off")

        # Panel 3: Prediction
        target_ax = axes[2]
    else:
        target_ax = axes[1]

    target_ax.imshow(mri_vol[best_slice_idx], cmap="gray")
    pred_overlay = np.ma.masked_where(~pred_vol[best_slice_idx], pred_vol[best_slice_idx])
    target_ax.imshow(pred_overlay, cmap="autumn", alpha=0.6)
    title_suffix = f" (Dice: {dice_score:.1f}%)" if dice_score is not None else ""
    target_ax.set_title(f"nnU-Net Prediction{title_suffix}", color="white", fontsize=14)
    target_ax.axis("off")

    preview_path = os.path.join(output_dir, "prediction_preview.png")
    plt.tight_layout()
    plt.savefig(preview_path, facecolor="black", dpi=150)
    plt.close()
    print(f"[✓] Visual preview saved:  {preview_path}\n")


# ==============================================================================
# S3 & CLASSIFICATION STUBS
# ==============================================================================
class S3Stub:
    """Simulates S3 MinIO storage locally."""
    def __init__(self, local_t1_path, local_t2_path=None):
        self.local_t1_path = local_t1_path
        self.local_t2_path = local_t2_path

    def load(self, path):
        # Serve T2 if requested, otherwise T1
        target_file = self.local_t2_path if ("t2" in path.lower() and self.local_t2_path) else self.local_t1_path
        print(f"[S3 Stub] Reading local file: {target_file}")
        with open(target_file, "rb") as f:
            return f.read()

    def store(self, obj, path, content_type):
        pass


class DummyClassificationModel:
    """Mocks Knosp classification so EfficientNet weights are not required."""
    def __init__(self, model_type="all"):
        pass

    def predict(self, rois: list) -> tuple:
        print("[Classification Stub] Generating mock Knosp probabilities...")
        individual_probs = []
        tracked_nodules_probs = {}

        for r in rois:
            slice_probs = []
            for nd in r:
                formation_id = nd[1]
                # Default high-probability Knosp 0-1-2 score
                mock_scores = [0.85, 0.10, 0.05]
                slice_probs.append(mock_scores)
                tracked_nodules_probs[formation_id] = mock_scores
            individual_probs.append(slice_probs)

        return individual_probs, tracked_nodules_probs


# ==============================================================================
# MAIN TEST PIPELINE
# ==============================================================================
def test_mri_pipeline():
    LOCAL_T1 = "./test_data/patient_T1.tif"
    LOCAL_T2 = "./test_data/patient_T2.tif"
    LOCAL_MASK = "./test_data/patient_T1_MASK.tif"

    if not os.path.exists(LOCAL_T1):
        print(f"Error: Input file {LOCAL_T1} not found.")
        return

    with tifffile.TiffFile(LOCAL_T1) as tif:
        num_slices = len(tif.pages)
    print(f"Found input scan with {num_slices} slices.")

    print("Initializing models...")
    seg_model = NnUNetSegmentationModel(model_dir=NNUNET_RESULTS_DIR)
    eff_model = DummyClassificationModel()
    s3_stub = S3Stub(LOCAL_T1, LOCAL_T2 if os.path.exists(LOCAL_T2) else None)

    # Wrap model prediction cleanly without altering production code
    original_predict = seg_model.predict

    def intercept_and_save(group):
        masks, rois = original_predict(group)
        save_and_evaluate(group, masks, output_dir="./test_data", gt_mask_path=LOCAL_MASK)
        return masks, rois

    seg_model.predict = intercept_and_save

    # Initialize business logic use-case
    mri_service = usecasemri.mriUseCase(seg_model, eff_model, s3_stub)

    # Intercept Kafka message
    mock_producer = MagicMock()
    usecasemri.Producer = MagicMock(return_value=mock_producer)

    print("Executing complete segmentClassificateSave pipeline...")
    pages_id = [f"slice_uuid_{i}" for i in range(num_slices)]
    mri_service.segmentClassificateSave(mri_id="test_case_001", pages_id=pages_id)

    # Validate output event structure
    if mock_producer.produce.called:
        topic = mock_producer.produce.call_args[0][0]
        raw_msg = mock_producer.produce.call_args[0][1]
        print(f"✓ Pipeline Finished Successfully!")
        print(f"✓ Protobuf event generated ({len(raw_msg)} bytes) for Kafka topic: '{topic}'")
    else:
        print("✓ Pipeline Completed (no nodules identified to publish).")


if __name__ == "__main__":
    test_mri_pipeline()
import os
import multiprocessing

try:
    multiprocessing.set_start_method('fork', force=True)
except RuntimeError:
    pass

os.environ.setdefault('S3_SECRET_KEY', 'dummy_secret_key')
os.environ.setdefault('S3_ACCESS_KEY', 'dummy_access_key')

# relative
settings = {
    'nnunet': './ml_service/ml_service/models/nnunet',
    'test_t1': './test_data/patient_T1.tif',
    'test_t2': './test_data/patient_T2.tif',
    'test_mask': './test_data/patient_T1_MASK.tif',
    'output_dir': './test_data'
}

os.environ['nnUNet_results'] = os.path.abspath(settings['nnunet'])
os.environ['nnUNet_raw'] = '/tmp/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/tmp/nnUNet_preprocessed'

import numpy as np
import tifffile
import nibabel as nib
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

from ml_service.internal.ml_model.nnunet_segmentation import NnUNetSegmentationModel
import ml_service.internal.usecases.mri.mri as usecasemri


def save_and_evaluate(parsed_slices: list, result_masks: list, output_dir: str, gt_mask_path: str = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    mri_vol = np.stack(parsed_slices, axis=0)
    pred_vol = np.stack(result_masks, axis=0) > 0
    num_slices = mri_vol.shape[0]

    tif_path = os.path.join(output_dir, 'predicted_mask.tif')
    tifffile.imwrite(tif_path, (pred_vol * 255).astype(np.uint8), compression='zlib')

    affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
    mask_nii = nib.Nifti1Image(np.transpose(pred_vol.astype(np.uint8), (1, 2, 0)), affine)
    nii_path = os.path.join(output_dir, 'predicted_mask.nii.gz')
    nib.save(mask_nii, nii_path)

    gt_vol = None
    dice_score = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_vol = tifffile.imread(gt_mask_path) > 0
        if gt_vol.shape == pred_vol.shape:
            intersection = 2.0 * np.sum(gt_vol & pred_vol)
            total = np.sum(gt_vol) + np.sum(pred_vol)
            dice_score = (intersection / total * 100.0) if total > 0 else 100.0

            print('Slice-by-slice evaluation:')
            print(f"{'Slice':<8} | {'GT voxels':<12} | {'Pred voxels':<12}")
            for i in range(num_slices):
                gt_cnt = int(np.sum(gt_vol[i]))
                pred_cnt = int(np.sum(pred_vol[i]))
                print(f'{i:<8} | {gt_cnt:<12} | {pred_cnt:<12}')
            print(f'Overall Dice: {dice_score:.2f}%')

    if gt_vol is not None:
        best_slice = int(np.argmax(np.sum(gt_vol, axis=(1, 2))))
    else:
        best_slice = int(np.argmax(np.sum(pred_vol, axis=(1, 2)))) if np.sum(pred_vol) > 0 else num_slices // 2

    has_gt = gt_vol is not None
    fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(18 if has_gt else 12, 6), facecolor='black')

    axes[0].imshow(mri_vol[best_slice], cmap='gray')
    axes[0].set_title(f'T1 slice {best_slice}', color='white')
    axes[0].axis('off')

    if has_gt:
        axes[1].imshow(mri_vol[best_slice], cmap='gray')
        gt_overlay = np.ma.masked_where(~gt_vol[best_slice], gt_vol[best_slice])
        axes[1].imshow(gt_overlay, cmap='winter', alpha=0.6)
        axes[1].set_title('Ground truth', color='white')
        axes[1].axis('off')
        target_ax = axes[2]
    else:
        target_ax = axes[1]

    target_ax.imshow(mri_vol[best_slice], cmap='gray')
    pred_overlay = np.ma.masked_where(~pred_vol[best_slice], pred_vol[best_slice])
    target_ax.imshow(pred_overlay, cmap='autumn', alpha=0.6)
    title_suffix = f' (Dice: {dice_score:.1f}%)' if dice_score is not None else ''
    target_ax.set_title(f'Prediction{title_suffix}', color='white')
    target_ax.axis('off')

    preview_path = os.path.join(output_dir, 'prediction_preview.png')
    plt.tight_layout()
    plt.savefig(preview_path, facecolor='black', dpi=150)
    plt.close()


class S3Stub:
    def __init__(self, local_t1_path: str, local_t2_path: str = None):
        self.local_t1_path = local_t1_path
        self.local_t2_path = local_t2_path

    def load(self, path: str) -> bytes:
        target = self.local_t2_path if ('t2' in path.lower() and self.local_t2_path) else self.local_t1_path
        with open(target, 'rb') as f:
            return f.read()

    def store(self, obj, path: str, content_type: str = None):
        pass


class DummyClassificationModel:
    def __init__(self, model_type: str = 'all'):
        self.model_type = model_type

    def predict(self, rois: list) -> tuple:
        """
        args:
            rois: массив узлов по срезам
        return:
            individual_probs, tracked_nodules_probs
        """
        individual_probs = []
        tracked_nodules_probs = {}

        for r in rois:
            slice_probs = []
            for nd in r:
                formation_id = nd[1]
                mock_scores = [0.85, 0.10, 0.05]
                slice_probs.append(mock_scores)
                tracked_nodules_probs[formation_id] = mock_scores
            individual_probs.append(slice_probs)

        return individual_probs, tracked_nodules_probs


def test_mri_pipeline():
    t1_path = settings['test_t1']
    t2_path = settings['test_t2']
    mask_path = settings['test_mask']
    output_dir = settings['output_dir']

    if not os.path.exists(t1_path):
        print(f'File not found: {t1_path}')
        return

    with tifffile.TiffFile(t1_path) as tif:
        num_slices = len(tif.pages)

    seg_model = NnUNetSegmentationModel(model_dir=settings['nnunet'])
    eff_model = DummyClassificationModel()
    s3_stub = S3Stub(t1_path, t2_path if os.path.exists(t2_path) else None)

    original_predict = seg_model.predict

    def intercept_and_save(group: list) -> tuple:
        masks, rois = original_predict(group)
        save_and_evaluate(group, masks, output_dir=output_dir, gt_mask_path=mask_path)
        return masks, rois

    seg_model.predict = intercept_and_save

    mri_service = usecasemri.mriUseCase(seg_model, eff_model, s3_stub)

    mock_producer = MagicMock()
    usecasemri.Producer = MagicMock(return_value=mock_producer)

    print('Pipeline execution...')
    pages_id = [f'slice_{i}' for i in range(num_slices)]
    mri_service.segmentClassificateSave(mri_id='test_case_001', pages_id=pages_id)
    print('Done!')


if __name__ == '__main__':
    test_mri_pipeline()
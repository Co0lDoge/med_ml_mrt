import os
import tempfile
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import label
from skimage.transform import resize
from ml_service.internal.ml_model.neuro_class import ModelABC

# Import native nnU-Net predictor
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class NnUNetSegmentationModel(ModelABC):
    def __init__(self, model_dir: str, fold: int = 0):
        super().__init__()
        self.model_dir = os.path.abspath(model_dir)
        self.fold = fold
        self.target_shape = (224, 224)

        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        os.environ['nnUNet_compile'] = 'False'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize predictor once
        print(f"Loading nnU-Net predictor from: {self.model_dir} on {self.device}...")
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        # Initializes from model directory without needing nnUNet_results env var
        self.predictor.initialize_from_trained_model_folder(
            self.model_dir,
            use_folds=(self.fold,),
            checkpoint_name="checkpoint_best.pth",
        )
        print("nnU-Net predictor initialized successfully!")

    def load(self, path: str = None) -> None:
        pass

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return image

    def _get_crop_bbox(self, original_shape, padding=48):
        z_c = original_shape[0] // 2
        y_c = original_shape[1] // 2
        x_c = original_shape[2] // 2
        return (
            slice(max(0, z_c - 4), min(original_shape[0], z_c + 4)),
            slice(max(0, y_c - padding), min(original_shape[1], y_c + padding)),
            slice(max(0, x_c - padding), min(original_shape[2], x_c + padding)),
        )

    def _save_nii(self, img_array, path):
        transposed = np.transpose(img_array, (1, 2, 0)).astype(np.float32)
        affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
        nib.save(nib.Nifti1Image(transposed, affine), path)

    def predict(self, group: list) -> tuple:
        # 1. Assemble 3D volume (Z, Y, X)
        volume_t1 = np.stack(group, axis=0).astype(np.float32)
        orig_shape = volume_t1.shape
        volume_t2 = volume_t1.copy()  # fallback if single modality

        # 2. Crop to ROI
        bbox = self._get_crop_bbox(orig_shape)
        crop_t1 = volume_t1[bbox]
        crop_t2 = volume_t2[bbox]

        # 3. Run prediction directly in Python
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            case_id = "case_001"
            self._save_nii(crop_t1, os.path.join(in_dir, f"{case_id}_0000.nii.gz"))
            self._save_nii(crop_t2, os.path.join(in_dir, f"{case_id}_0001.nii.gz"))

            # Native predictor call
            self.predictor.predict_from_files(
                in_dir,
                out_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )

            pred_file = os.path.join(out_dir, f"{case_id}.nii.gz")
            if os.path.exists(pred_file):
                pred_crop = nib.load(pred_file).get_fdata()
                pred_crop = np.transpose(pred_crop, (2, 0, 1))
            else:
                pred_crop = np.zeros(crop_t1.shape, dtype=np.uint8)

        # 4. Post-processing: keep largest connected component
        labeled, num = label(pred_crop > 0)
        if num > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            pred_crop = (labeled == sizes.argmax()).astype(np.uint8)

        # 5. Uncrop
        full_mask = np.zeros(orig_shape, dtype=np.uint8)
        full_mask[bbox] = (pred_crop > 0).astype(np.uint8)

        # 6. Format result_masks and rois
        result_masks = []
        rois = []

        for z in range(orig_shape[0]):
            slice_mask = full_mask[z]
            slice_img = volume_t1[z]
            result_masks.append((slice_mask * 255).astype(np.uint8))

            slice_rois = []
            if np.sum(slice_mask) > 0:
                coords = np.argwhere(slice_mask > 0)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                roi_crop = slice_img[max(0, y_min - 10):min(slice_img.shape[0], y_max + 10),
                                     max(0, x_min - 10):min(slice_img.shape[1], x_max + 10)]
                roi_crop = resize(roi_crop, self.target_shape, order=1, preserve_range=True)
                slice_rois.append([roi_crop, 0, slice_mask.astype(np.float32)])

            rois.append(slice_rois)

        return result_masks, rois
# ml_service/internal/ml_model/nnunet_segmentation.py

import os
import tempfile
import numpy as np
import nibabel as nib
import tifffile
import torch
import torch.nn.functional as F
from ml_service.internal.ml_model.neuro_class import ModelABC
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

try:
    import cc3d
except ImportError:
    cc3d = None


def resize_array(img_array: np.ndarray, target_shape: tuple, is_mask: bool = False) -> np.ndarray:
    """
    Handles both 2D slices (bilinear) and 3D volumes (trilinear) natively in PyTorch.
    """
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()

    if img_array.ndim == 2:
        mode = 'nearest' if is_mask else 'bilinear'
    elif img_array.ndim == 3:
        mode = 'nearest' if is_mask else 'trilinear'
    else:
        raise ValueError(f"Expected 2D or 3D numpy array, got {img_array.ndim}D")

    align_corners = None if is_mask else False
    resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=align_corners)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.uint8 if is_mask else np.float32)


class NnUNetSegmentationModel(ModelABC):
    def __init__(self, model_dir: str, fold: int = 0, template_mask_path: str = None):
        super().__init__()
        self.model_dir = os.path.abspath(model_dir)
        self.fold = fold
        self.target_shape = (224, 224)
        
        # ----------------------------------------------------------------------
        # 1. OPTION 2: LOAD GOLDEN REFERENCE MASK
        # ----------------------------------------------------------------------
        if template_mask_path is None:
            # Default location inside the model directory
            template_mask_path = os.path.join(self.model_dir, "template_mask.tif")

        self.roi_normalized = None  # Stored as relative ratios (0.0 to 1.0)
        self._init_golden_reference(template_mask_path)

        # ----------------------------------------------------------------------
        # 2. ENVIRONMENT & PREDICTOR INITIALIZATION
        # ----------------------------------------------------------------------
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        os.environ['nnUNet_compile'] = 'False'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[NnUNet] Initializing predictor from: {self.model_dir} on {self.device}...")

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,  # Matches CLI !nnUNetv2_predict (TTA)
            perform_everything_on_device=(self.device.type == 'cuda'),
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        self.predictor.initialize_from_trained_model_folder(
            self.model_dir, 
            use_folds=(self.fold,), 
            checkpoint_name="checkpoint_best.pth"
        )
        print("[NnUNet] Predictor ready.")

    def _init_golden_reference(self, path: str, padding: int = 48) -> None:
        """Parses the Golden Reference Mask once at startup."""
        if not os.path.exists(path):
            print(f"[!] Warning: Golden reference mask not found at: {path}")
            print("    Using default anatomical fallback (Y: ~38%-70%, X: ~32%-68%).")
            self.roi_normalized = (0.38, 0.70, 0.32, 0.68)
            return

        print(f"[✓] Loading Golden Reference Mask from: {path}")
        ref_mask = tifffile.imread(path) > 0
        coords = np.argwhere(ref_mask)

        if len(coords) == 0:
            raise ValueError(f"Golden reference mask at {path} is empty (all zeros).")

        # Handle both 2D (Y, X) and 3D (Z, Y, X)
        if ref_mask.ndim == 3:
            h, w = ref_mask.shape[1], ref_mask.shape[2]
            y_coords, x_coords = coords[:, 1], coords[:, 2]
        else:
            h, w = ref_mask.shape[0], ref_mask.shape[1]
            y_coords, x_coords = coords[:, 0], coords[:, 1]

        y_min, y_max = max(0, y_coords.min() - padding), min(h, y_coords.max() + padding)
        x_min, x_max = max(0, x_coords.min() - padding), min(w, x_coords.max() + padding)

        # Store as resolution-independent fractions (e.g. 0.42 to 0.66)
        self.roi_normalized = (y_min / h, y_max / h, x_min / w, x_max / w)
        print(f"[✓] Golden Reference ROI initialized: Y=[{y_min}:{y_max}], X=[{x_min}:{x_max}] (normalized: {np.round(self.roi_normalized, 3)})")

    def _get_crop_bbox(self, original_shape: tuple) -> tuple:
        """
        Derives the 3D bounding box for any incoming scan:
        - Z: Takes 100% of slices (never truncates Z).
        - Y, X: Projects the golden reference template onto the scan's resolution.
        """
        num_slices, h, w = original_shape[0], original_shape[1], original_shape[2]

        # 1. Z-axis: Always take ALL slices
        z_slice = slice(0, num_slices)

        # 2. In-plane: Scale normalized golden template to current scan dimensions
        y_norm_min, y_norm_max, x_norm_min, x_norm_max = self.roi_normalized
        y_min, y_max = int(y_norm_min * h), int(y_norm_max * h)
        x_min, x_max = int(x_norm_min * w), int(x_norm_max * w)

        return (z_slice, slice(y_min, y_max), slice(x_min, x_max))

    def _save_nii(self, arr: np.ndarray, path: str) -> None:
        """Transposes (Z, Y, X) to (Y, X, Z) for nnU-Net's convention."""
        transposed = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
        nib.save(nib.Nifti1Image(transposed, affine), path)

    def _keep_largest_component(self, binary_mask: np.ndarray) -> np.ndarray:
        """26-connectivity connected components to keep multi-lobular lesions intact."""
        data = (binary_mask > 0).astype(np.uint8)
        if cc3d is not None:
            labels_out, N = cc3d.connected_components(data, connectivity=26, return_N=True)
            if N > 1:
                sizes = np.bincount(labels_out.ravel())
                sizes[0] = 0
                return labels_out == sizes.argmax()
            return data > 0
        else:
            from scipy.ndimage import label
            labeled, num = label(data > 0)
            if num > 1:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                return labeled == sizes.argmax()
            return data > 0

    def load(self, path: str = None) -> None:
        pass

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return image

    def predict(self, group: list, group_t2: list = None) -> tuple:
        # 1. Assemble T1 (Z, Y, X)
        vol_t1 = np.stack(group, axis=0).astype(np.float32)
        orig_shape = vol_t1.shape

        # 2. Assemble & align T2
        if group_t2 is not None:
            vol_t2 = np.stack(group_t2, axis=0).astype(np.float32)
            if vol_t2.shape != orig_shape:
                vol_t2 = resize_array(vol_t2, orig_shape, is_mask=False)
        else:
            vol_t2 = vol_t1.copy()

        # 3. Crop using the Golden Reference ROI (All Z, Golden Y & X)
        bbox = self._get_crop_bbox(orig_shape)
        crop_t1 = vol_t1[bbox]
        crop_t2 = vol_t2[bbox]

        # 4. Predict via nnU-Net
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            case_id = "case_001"
            self._save_nii(crop_t1, os.path.join(in_dir, f"{case_id}_0000.nii.gz"))
            self._save_nii(crop_t2, os.path.join(in_dir, f"{case_id}_0001.nii.gz"))

            self.predictor.predict_from_files(
                in_dir, 
                out_dir, 
                save_probabilities=False, 
                overwrite=True,
                num_processes_preprocessing=1, 
                num_processes_segmentation_export=1
            )

            pred_crop_nii = nib.load(os.path.join(out_dir, f"{case_id}.nii.gz")).get_fdata()
            pred_crop = np.transpose(pred_crop_nii, (2, 0, 1)) > 0

        # 5. Un-crop into full volume dimensions & clean isolated noise
        full_mask = np.zeros(orig_shape, dtype=bool)
        full_mask[bbox] = pred_crop
        full_mask = self._keep_largest_component(full_mask)

        # 6. Build contract for mri.py
        result_masks = []
        rois = []

        for z in range(orig_shape[0]):
            slice_mask = full_mask[z]
            slice_img = vol_t1[z]
            result_masks.append((slice_mask * 255).astype(np.uint8))

            slice_rois = []
            if np.sum(slice_mask) > 0:
                coords = np.argwhere(slice_mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                roi_crop = slice_img[
                    max(0, y_min - 10):min(slice_img.shape[0], y_max + 10),
                    max(0, x_min - 10):min(slice_img.shape[1], x_max + 10)
                ]
                roi_crop = resize_array(roi_crop, self.target_shape)
                slice_rois.append([roi_crop, 0, slice_mask.astype(np.float32)])

            rois.append(slice_rois)

        return result_masks, rois
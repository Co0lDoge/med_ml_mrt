import os
import tempfile
import numpy as np
import nibabel as nib
import torch
from torch.nn import functional as F
from ml_service.internal.ml_model.neuro_class import ModelABC
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

try:
    import cc3d
except ImportError:
    cc3d = None

# relative
settings = {
    'nnunet': './ml_service/ml_service/models/nnunet'
}


def resize_array(img_array: np.ndarray, target_shape: tuple, is_mask: bool = False) -> np.ndarray:
    """Интерполяция 2D срезов (bilinear) и 3D объемов (trilinear) через PyTorch."""
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

    def __init__(self, model_dir: str = None, fold: int = 0):
        super().__init__()
        self.model_dir = model_dir if model_dir else settings['nnunet']
        self.fold = fold
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.target_shape = (224, 224)

        # Жестко заданный ROI (Y=[180:341], X=[177:351] для 512x512)
        # Сохранен в виде относительных долей [y_min, y_max, x_min, x_max] от исходного разрешения
        self.roi_normalized = (0.352, 0.666, 0.346, 0.686)

        # Флаги окружения для стабильного инференса
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        os.environ['nnUNet_compile'] = 'False'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'

        self.load(self.model_dir)

    def load(self, path: str) -> None:
        self._model_dir = os.path.abspath(path)
        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=(self.device.type == 'cuda'),
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        self._predictor.initialize_from_trained_model_folder(
            self._model_dir,
            use_folds=(self.fold,),
            checkpoint_name="checkpoint_best.pth"
        )

    def preprocessing(self, image: np.ndarray) -> np.ndarray:
        return image

    def _get_crop_bbox(self, original_shape: tuple) -> tuple:
        """
        Формирует 3D bounding box для обрезки:
        """
        num_slices, h, w = original_shape[0], original_shape[1], original_shape[2]

        z_slice = slice(0, num_slices)
        y_min, y_max = int(self.roi_normalized[0] * h), int(self.roi_normalized[1] * h)
        x_min, x_max = int(self.roi_normalized[2] * w), int(self.roi_normalized[3] * w)

        return (z_slice, slice(y_min, y_max), slice(x_min, x_max))

    @staticmethod
    def _save_nii(arr: np.ndarray, path: str) -> None:
        transposed = np.transpose(arr, (1, 2, 0)).astype(np.float32)
        affine = np.diag([0.5078, 0.5078, 3.3, 1.0])
        nib.save(nib.Nifti1Image(transposed, affine), path)

    @staticmethod
    def _keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
        data = (binary_mask > 0).astype(np.uint8)
        if cc3d is not None:
            labels_out, n = cc3d.connected_components(data, connectivity=26, return_N=True)
            if n > 1:
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

    def predict(self, group: list, group_t2: list = None) -> tuple:
        """
        args:
            group: list из 2D numpy arrays для снимков T1 (длина списка = кол-во срезов)
            group_t2: list из 2D numpy arrays для снимков T2

        return:
            result_masks - список 2D uint8 бинарных масок [0, 255] по всем срезам
            rois - массив по изображениям:
                [
                    [
                        [scaled_pic_of_roi, roi_uniq_idx, segment_binary_masks]
                    ]
                ]
        """
        print(f'Device: {self.device}')
        print('Segmentation inference...')

        vol_t1 = np.stack(group, axis=0).astype(np.float32)
        orig_shape = vol_t1.shape

        if group_t2 is not None:
            vol_t2 = np.stack(group_t2, axis=0).astype(np.float32)
            if vol_t2.shape != orig_shape:
                vol_t2 = resize_array(vol_t2, orig_shape, is_mask=False)
        else:
            vol_t2 = vol_t1.copy()

        # Статический кроп ROI
        bbox = self._get_crop_bbox(orig_shape)
        crop_t1 = vol_t1[bbox]
        crop_t2 = vol_t2[bbox]

        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            case_id = "case_001"
            self._save_nii(crop_t1, os.path.join(in_dir, f"{case_id}_0000.nii.gz"))
            self._save_nii(crop_t2, os.path.join(in_dir, f"{case_id}_0001.nii.gz"))

            self._predictor.predict_from_files(
                in_dir,
                out_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1
            )

            pred_crop_nii = nib.load(os.path.join(out_dir, f"{case_id}.nii.gz")).get_fdata()
            pred_crop = np.transpose(pred_crop_nii, (2, 0, 1)) > 0

        # Возврат в исходные координаты и фильтрация _keep_largest_component
        full_mask = np.zeros(orig_shape, dtype=bool)
        full_mask[bbox] = pred_crop
        full_mask = self._keep_largest_component(full_mask)

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

        print('Done!')
        return result_masks, rois
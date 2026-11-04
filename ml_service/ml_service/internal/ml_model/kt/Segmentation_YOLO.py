import numpy as np
import warnings
import cv2
from ultralytics import YOLO
import tempfile

warnings.filterwarnings("ignore")


def video_to_npy_channel3(video_data, target_frames=54, target_size=(224, 224)):
    # Считываем все доступные кадры
    frames = get_frames_from_video_data(video_data)

    total_frames = len(frames)
    if total_frames < target_frames:
        raise ValueError(f"Видео содержит только {total_frames} кадров. Требуется минимум {target_frames}.")

    # Равномерно выбираем целевые кадры
    indices = np.linspace(0, total_frames - 1, num=target_frames, dtype=int)
    selected_frames = []

    for idx in indices:
        frame = frames[idx]
        # Конвертируем BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Изменяем размер
        frame_resized = cv2.resize(frame_rgb, target_size)
        selected_frames.append(frame_resized)

    # Создаем массив NumPy
    np_array = np.array(selected_frames, dtype=np.uint8)

    # Проверяем размерность
    if np_array.shape != (target_frames, *target_size, 3):
        raise RuntimeError(f"Ошибка размерности. Получено: {np_array.shape}")
    
    return np_array


class SegmentationModel:
    def __init__(self) -> None:
        import os
        model_path = "./ml_service/internal/ml_model/models/kt/yolo_seg_small_040325.pt"
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"[WARN] YOLO segmentation weights not found at {model_path}.")
            self.model = None
    
    
    def predict(self, video_bytes,  target_frames=54, target_size=(224, 224), 
                conf_threshold=0.5, mask_threshold=0.5, detection_color=(0, 255, 0)):
        """
        Метод конвертирует видео в numpy array, обрабатывает его с YOLO моделью и возвращает массивы с кадрами и массив с ROI.
        ROI - List[List[List[List[x1, y1], List[x2, y1], List[x1, y2], List[x2, y2]]]]
        [ - Список изображений
            [ - Список узлов на изображении (в КТ один)
                [ - Список вершин контура
                    [x1, y1], [x2, y1], [x1, y2], [x2, y2] - Координаты вершин
                ]
            ]
        ]
        """
        # Конвертация видео в numpy array
        video_array = self._video_bytes_to_npy(video_bytes, target_frames, target_size)

        # Обработка с YOLO моделью
        original_frames, mask_frames, detected_frames, rois_in_frames = self._process_with_yolo(video_array, conf_threshold, mask_threshold, detection_color)
        
        return {"original": original_frames,
                "mask": mask_frames,
                "detection": detected_frames,
                "rois": rois_in_frames}
    
    def _video_bytes_to_npy(self, video_bytes, target_frames, target_size):
        frames = []
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.mp4') as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()  # Убедимся, что данные записаны на диск
            
            # Открываем видео с помощью OpenCV
            cap = cv2.VideoCapture(temp_file.name)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            total_frames = len(frames)
            if total_frames < target_frames:
                raise ValueError(f"Видео содержит только {total_frames} кадров. Требуется минимум {target_frames}.")
            
            indices = np.linspace(0, total_frames - 1, num=target_frames, dtype=int)
            selected_frames = []

            for idx in indices:
                frame = frames[idx]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, target_size)
                selected_frames.append(frame_resized)
            
        return np.array(selected_frames, dtype=np.uint8)
    
    def _process_with_yolo(self, video_array, conf_threshold, mask_threshold, detection_color):
        H, W = video_array.shape[1], video_array.shape[2]
        """Обработка видео с YOLO моделью"""
        original_frames = []
        mask_frames = []
        detected_frames = []

        rois_in_frames = []             # List[List[List[List[x1, y1], List[x2, y1], List[x1, y2], List[x2, y2]]]]
                                        # Для каждого кадра берем список всех его ROI
                                        # Для каждого ROI берем списки X и Y координат
                                        # координаты: [левый верхний угол, правый верхний угол, левый нижний угол, правый нижний угол]

        
        for frame in video_array:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = self.model(frame_bgr, conf=conf_threshold, task="segment", verbose=False)[0]
            # оригинал
            original_frames.append(frame)
            
            # маска
            H, W = frame.shape[:2]
            mask_output = np.zeros((H, W), dtype=np.uint8)
            
            if results.masks is not None and results.masks.data.numel() > 0:
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    binary = (mask > mask_threshold).astype(np.uint8) * 255
                    mask_output = np.maximum(mask_output, binary)
            
            mask_frames.append(mask_output)

            # видео с детекцией и roi
            frame_rois = []
            det_frame = frame_bgr.copy()
            for box in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box[:6]
                if conf < conf_threshold:
                    continue

                # Добавляем 1 кадр (координаты вершин прямоугольника)
                frame_rois.append([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
                
                # Рисуем bounding box
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(det_frame, pt1, pt2, detection_color, 2)
                
                # Добавляем текст
                label = f"{results.names[int(cls)]} {conf:.2f}"
                cv2.putText(det_frame, label, (pt1[0], pt1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, detection_color, 2)
            
            # Добавляем кадры roi
            rois_in_frames.append(frame_rois)

            # Конвертируем обратно в RGB для согласованности
            det_frame_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            detected_frames.append(det_frame_rgb)

        
        return np.array(original_frames), np.array(mask_frames), np.array(detected_frames), rois_in_frames
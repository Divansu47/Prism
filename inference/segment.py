from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F


class FoodSegmenter:

    ROI_PADDING = 0.15
    MAX_ROI_RATIO = 0.85

    MIN_COMPONENT_AREA = 100

    SMALL_KERNEL = 3
    MEDIUM_KERNEL = 5
    LARGE_KERNEL = 7

    def __init__(self, model_path=None):

        root = Path(__file__).resolve().parents[1]
        #older version
        # if model_path is None:
        #     model_path = (
        #         root
        #         / "runs"
        #         / "segment"
        #         / "runs"
        #         / "segment"
        #         / "foodseg103_final"
        #         / "weights"
        #         / "best.pt"
        #     )
        if model_path is None:
            # model_path = (
            #     root
            #     / "runs"
            #     / "segment"
            #     / "runs"
            #     / "segment"
            #     / "runs"
            #     / "foodseg103_rebalanced"
            #     / "weights"
            #     / "best.pt"
            model_path = root / "models" / "foodseg_best.pt"
            

        print("\n====================================")
        print("Loading model:")
        print(model_path)
        print("====================================\n")

        self.model = YOLO(str(model_path))

    def _run_prediction(self, image):

        return self.model.predict(
            source=image,
            conf=0.50,
            iou=0.70,
            max_det=300,
            verbose=False,
            augment=True, # Added TTA
        )[0]

    def _score_result(self, result):

        if len(result.boxes) == 0:
            return -1.0

        confs = result.boxes.conf.cpu().numpy()

        mean_conf = float(confs.mean())
        max_conf = float(confs.max())
        num_boxes = len(confs)

        score = (
            0.55 * mean_conf
            + 0.35 * max_conf
            + 0.10 * min(num_boxes / 10.0, 1.0)
        )

        return score

    def _kernel_size(self, area):

        if area < 5000:
            return self.SMALL_KERNEL

        if area < 25000:
            return self.MEDIUM_KERNEL

        return self.LARGE_KERNEL

    def _get_food_roi(self, image, result):

        if len(result.boxes) == 0:
            return None

        h, w = image.shape[:2]

        boxes = result.boxes.xyxy.cpu().numpy()

        x1 = float(boxes[:, 0].min())
        y1 = float(boxes[:, 1].min())
        x2 = float(boxes[:, 2].max())
        y2 = float(boxes[:, 3].max())

        bw = x2 - x1
        bh = y2 - y1

        pad_x = bw * self.ROI_PADDING
        pad_y = bh * self.ROI_PADDING

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(w, int(x2 + pad_x))
        y2 = min(h, int(y2 + pad_y))

        roi_area = (x2 - x1) * (y2 - y1)
        image_area = h * w

        if roi_area >= self.MAX_ROI_RATIO * image_area:
            return None

        return (x1, y1, x2, y2)

    def _analyze_roi(self, image, result):

        roi = self._get_food_roi(image, result)

        if roi is None:
            return result

        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        refined = self._run_prediction(crop)

        original_score = self._score_result(result)
        refined_score = self._score_result(refined)

        if refined_score > original_score:
            print(f"\n========== ROI ANALYSIS ==========")
            print("Applying refined ROI prediction.")
            print("==================================\n")

            # 1. Shift bounding boxes back to original coordinates
            refined.boxes.xyxy[:, 0] += x1
            refined.boxes.xyxy[:, 1] += y1
            refined.boxes.xyxy[:, 2] += x1
            refined.boxes.xyxy[:, 3] += y1

            # 2. Pad masks back to the original image dimensions
            if refined.masks is not None:
                pad_left = x1
                pad_right = image.shape[1] - x2
                pad_top = y1
                pad_bottom = image.shape[0] - y2
                
                refined.masks.data = F.pad(
                    refined.masks.data,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0
                )

            # 3. Restore original image metadata
            refined.orig_img = image
            refined.orig_shape = image.shape[:2]
            
            return refined

        return result

        print("==================================\n")
    def _refine_masks(self, result):

            if result.masks is None:
                return None

            refined_masks = []

            print("\n========== MASK REFINEMENT ==========")

            for i, mask in enumerate(result.masks.data.cpu().numpy()):

                confidence = float(result.boxes.conf[i])

                # Adaptive threshold based on confidence
                threshold = np.clip(
                    0.45 + (confidence - 0.50) * 0.20,
                    0.40,
                    0.50,
                )
                binary = (mask > threshold).astype(np.uint8)

                original_area = int(binary.sum())

                kernel_size = self._kernel_size(original_area)

                kernel = np.ones(
                    (kernel_size, kernel_size),
                    dtype=np.uint8,
                )

                # Fill small holes
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    kernel,
                )

                # Remove tiny noisy blobs
                binary = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_OPEN,
                    kernel,
                )

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary,
                    connectivity=8,
                )

                cleaned = np.zeros_like(binary)

                for label in range(1, num_labels):

                    area = stats[label, cv2.CC_STAT_AREA]

                    if area >= self.MIN_COMPONENT_AREA:
                        cleaned[labels == label] = 1

                refined_area = int(cleaned.sum())

                cls = int(result.boxes.cls[i])
                food = result.names[cls]

                print(
                    f"{food:20s}"
                    f"{original_area:7d} -> "
                    f"{refined_area:7d}"
                )

                refined_masks.append(cleaned.astype(bool))

            print("====================================\n")

            return refined_masks

    def segment_image(self, image):

        result = self._run_prediction(image)
        # Capture the output (it will return original if ROI isn't better)
        result = self._analyze_roi(image, result)

        # Refine masks for downstream area / volume estimation
        refined_masks = self._refine_masks(result)

        # Analyze ROI (does NOT modify detections)
        self._analyze_roi(image, result)

        # Refine masks for downstream area / volume estimation
        refined_masks = self._refine_masks(result)

        if refined_masks is not None:
            result.refined_masks = refined_masks

        print("\n========== YOLO DEBUG ==========")
        print("Number of boxes :", len(result.boxes))

        if len(result.boxes):

            confs = result.boxes.conf.cpu().numpy()

            print("Max confidence :", confs.max())
            print("Min confidence :", confs.min())

            print("\nTop 20 confidences:")
            print(np.sort(confs)[::-1][:20])

            print("\nTop detections:")

            order = np.argsort(confs)[::-1]

            for idx in order[:20]:

                cls = int(result.boxes.cls[idx])
                name = result.names[cls]

                print(
                    f"{name:20s} "
                    f"{confs[idx]:.4f}"
                )

        print("================================\n")

        return result

    def predict(self, image_path):

        results = self.model.predict(
            source=str(image_path),
            conf=0.50,
            iou=0.70,
            max_det=300,
            verbose=False,
            save=True,
        )

        result = results[0]

        if len(result.boxes) == 0:
            return None

        idx = int(np.argmax(result.boxes.conf.cpu().numpy()))

        cls = int(result.boxes.cls[idx])

        return {
            "food": result.names[cls],
            "confidence": float(result.boxes.conf[idx]),
        }
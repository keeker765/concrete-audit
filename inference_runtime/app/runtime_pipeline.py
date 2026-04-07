from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from concrete_match_model import ConcreteEarlyFusionNet


LEFT_ONLY_COLOR = (0, 180, 255)
RIGHT_ONLY_COLOR = (255, 120, 0)
OVERLAP_COLOR = (0, 220, 0)


def encode_jpg_base64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("图片编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def decode_upload(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取上传图片: {file_storage.filename}")
    return image


def build_rectify_matrix(
    points: np.ndarray,
    canvas_size: int,
    qr_anchor_size: int,
    qr_anchor_offset: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    src = points.reshape(4, 2).astype(np.float32)
    dst = np.array(
        [
            [qr_anchor_offset, qr_anchor_offset],
            [qr_anchor_offset + qr_anchor_size, qr_anchor_offset],
            [qr_anchor_offset + qr_anchor_size, qr_anchor_offset + qr_anchor_size],
            [qr_anchor_offset, qr_anchor_offset + qr_anchor_size],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst), (canvas_size, canvas_size)


def warp_mask(mask: np.ndarray, matrix: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    warped = cv2.warpPerspective(
        mask,
        matrix,
        out_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 127).astype(np.uint8) * 255


def warp_rgb(image: np.ndarray, matrix: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        matrix,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary * 255
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8) * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape[:2]
    flood = padded.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def refine_mask(mask: np.ndarray) -> np.ndarray:
    return keep_largest_component(fill_holes(keep_largest_component(mask)))


def create_overlay(image: np.ndarray, mask: np.ndarray, qr_points: np.ndarray | None) -> np.ndarray:
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 1.0, green, 0.45, 0)
    if qr_points is not None:
        pts = np.asarray(qr_points).reshape(4, 2).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 3)
    return overlay


def select_best_mask(result, conf_thres: float, image_shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise ValueError("YOLO 未检测到目标")
    confidences = result.boxes.conf.detach().cpu().numpy()
    best_index = int(np.argmax(confidences))
    best_conf = float(confidences[best_index])
    if best_conf < conf_thres:
        raise ValueError(f"最高置信度过低: {best_conf:.4f}")
    mask_tensor = result.masks.data[best_index].detach().cpu().numpy()
    mask = (mask_tensor > 0.5).astype(np.uint8) * 255
    image_height, image_width = image_shape
    if mask.shape != (image_height, image_width):
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
    return mask, best_conf


class QrDetector:
    def __init__(self) -> None:
        if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
            raise RuntimeError("当前环境缺少 cv2.wechat_qrcode_WeChatQRCode")
        self.detector = cv2.wechat_qrcode_WeChatQRCode()

    def detect(self, image: np.ndarray) -> tuple[str, np.ndarray] | None:
        texts, points = self.detector.detectAndDecode(image)
        if texts:
            return str(texts[0]), np.asarray(points[0], dtype=np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        texts, points = self.detector.detectAndDecode(enhanced)
        if texts:
            return str(texts[0]), np.asarray(points[0], dtype=np.float32)

        height, width = image.shape[:2]
        for scale in [0.5, 0.75, 1.5, 0.25]:
            resized = cv2.resize(image, (int(width * scale), int(height * scale)))
            texts, points = self.detector.detectAndDecode(resized)
            if texts:
                return str(texts[0]), np.asarray(points[0], dtype=np.float32) / scale
        return None


def binary_mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


def add_title(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return canvas


def fit_height(image: np.ndarray, target_h: int = 320) -> np.ndarray:
    scale = target_h / image.shape[0]
    interpolation = cv2.INTER_AREA if image.ndim == 3 else cv2.INTER_NEAREST
    return cv2.resize(image, (int(image.shape[1] * scale), target_h), interpolation=interpolation)


def build_overlap_image(left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    overlap = left_mask & right_mask
    left_only = left_mask & ~right_mask
    right_only = right_mask & ~left_mask
    canvas = np.zeros((*left_mask.shape, 3), dtype=np.uint8)
    canvas[left_only] = LEFT_ONLY_COLOR
    canvas[right_only] = RIGHT_ONLY_COLOR
    canvas[overlap] = OVERLAP_COLOR
    return canvas


def build_overlap_legend(width: int) -> np.ndarray:
    legend = np.full((56, width, 3), 22, dtype=np.uint8)
    entries = [("overlap", OVERLAP_COLOR), ("left_only", LEFT_ONLY_COLOR), ("right_only", RIGHT_ONLY_COLOR)]
    x = 12
    for text, color in entries:
        cv2.rectangle(legend, (x, 16), (x + 22, 38), color, thickness=-1)
        cv2.putText(legend, text, (x + 30, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        x += 190
    return legend


def build_pair_panel_from_arrays(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    title: str,
    subtitle: str,
) -> np.ndarray:
    left_binary = left_mask > 127
    right_binary = right_mask > 127
    overlap = build_overlap_image(left_binary, right_binary)
    left_rgb_vis = fit_height(add_title(left_rgb, "LEFT warped_rgb", (0, 255, 0)))
    right_rgb_vis = fit_height(add_title(right_rgb, "RIGHT warped_rgb", (0, 200, 255)))
    left_mask_vis = fit_height(add_title(binary_mask_to_bgr(left_binary), "LEFT mask", (0, 255, 0)))
    right_mask_vis = fit_height(add_title(binary_mask_to_bgr(right_binary), "RIGHT mask", (0, 200, 255)))
    overlap_vis = fit_height(add_title(overlap, "MASK overlap", (180, 220, 255)))
    top = np.hstack([left_rgb_vis, right_rgb_vis])
    bottom = np.hstack([left_mask_vis, right_mask_vis, overlap_vis])
    width = max(top.shape[1], bottom.shape[1])
    if top.shape[1] < width:
        top = np.hstack([top, np.zeros((top.shape[0], width - top.shape[1], 3), dtype=np.uint8)])
    if bottom.shape[1] < width:
        bottom = np.hstack([bottom, np.zeros((bottom.shape[0], width - bottom.shape[1], 3), dtype=np.uint8)])
    header = np.full((72, width, 3), 18, dtype=np.uint8)
    cv2.putText(header, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 210, 255), 2, cv2.LINE_AA)
    legend = build_overlap_legend(width)
    return np.vstack([header, top, bottom, legend])


def prepare_matcher_input(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = (mask_a > 127).astype(np.float32)
    b = (mask_b > 127).astype(np.float32)
    return np.stack([a, b], axis=0)[None, ...]


@dataclass(frozen=True)
class VerifyArtifacts:
    score: float
    pred: int
    verdict: str
    reason: str
    left_yolo_conf: float
    right_yolo_conf: float
    left_qr_text: str
    right_qr_text: str
    left_overlay_b64: str
    right_overlay_b64: str
    left_warped_b64: str
    right_warped_b64: str
    left_mask_b64: str
    right_mask_b64: str
    panel_b64: str
    threshold: float


class SinglePairPtVerifier:
    def __init__(
        self,
        matcher_weights: Path,
        yolo_weights: Path,
        yolo_device: str,
        imgsz: int,
        conf_thres: float,
        warp_canvas_size: int,
        qr_anchor_size: int,
        qr_anchor_offset: int,
        threshold: float,
    ) -> None:
        if not matcher_weights.exists():
            raise FileNotFoundError(f"matcher 权重不存在: {matcher_weights}")
        if not yolo_weights.exists():
            raise FileNotFoundError(f"YOLO 权重不存在: {yolo_weights}")
        self.yolo_device = yolo_device
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.warp_canvas_size = warp_canvas_size
        self.qr_anchor_size = qr_anchor_size
        self.qr_anchor_offset = qr_anchor_offset
        self.threshold = threshold
        self.segmenter = YOLO(str(yolo_weights))
        self.qr_detector = QrDetector()
        self.device = torch.device("cpu")
        self.matcher = ConcreteEarlyFusionNet().to(self.device)
        checkpoint = torch.load(matcher_weights, map_location=self.device)
        self.matcher.load_state_dict(checkpoint["model_state_dict"])
        self.matcher.eval()

    def _prepare(self, image: np.ndarray, side_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
        prediction = self.segmenter.predict(
            source=image,
            imgsz=self.imgsz,
            conf=0.01,
            device=self.yolo_device,
            verbose=False,
            retina_masks=True,
        )[0]
        try:
            mask, yolo_conf = select_best_mask(prediction, self.conf_thres, image.shape[:2])
        except ValueError as exc:
            message = str(exc)
            if "最高置信度过低" in message:
                raise ValueError(f"{side_name} 不能有效提取边框，YOLO 最高置信度过低: {message.split(':', 1)[-1].strip()}") from exc
            if "YOLO 未检测到目标" in message:
                raise ValueError(f"{side_name} 不能有效提取边框，YOLO 未检测到目标") from exc
            raise
        mask = refine_mask(mask)
        qr_result = self.qr_detector.detect(image)
        if qr_result is None:
            raise ValueError(f"{side_name} 二维码未知，无法完成校正")
        qr_text, qr_points = qr_result
        matrix, warped_size = build_rectify_matrix(
            qr_points,
            canvas_size=self.warp_canvas_size,
            qr_anchor_size=self.qr_anchor_size,
            qr_anchor_offset=self.qr_anchor_offset,
        )
        warped_rgb = warp_rgb(image, matrix, warped_size)
        warped_mask = refine_mask(warp_mask(mask, matrix, warped_size))
        overlay = create_overlay(image, mask, qr_points)
        return overlay, warped_rgb, warped_mask, yolo_conf, qr_text

    def verify(self, left_image: np.ndarray, right_image: np.ndarray) -> VerifyArtifacts:
        left_overlay, left_warped, left_mask, left_yolo_conf, left_qr_text = self._prepare(left_image, "左图")
        right_overlay, right_warped, right_mask, right_yolo_conf, right_qr_text = self._prepare(right_image, "右图")
        if left_qr_text != right_qr_text:
            score = 0.03
            pred = 0
            verdict = "更偏向假"
            reason = "二维码不一致。"
        else:
            inputs = prepare_matcher_input(left_mask, right_mask)
            tensor = torch.from_numpy(inputs).to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                logits = self.matcher(tensor)
                score = float(torch.sigmoid(logits).item())
            pred = 1 if score >= self.threshold else 0
            verdict = "更偏向真" if pred == 1 else "更偏向假"
            reason = "二维码一致。"
        panel = build_pair_panel_from_arrays(
            left_warped,
            right_warped,
            left_mask,
            right_mask,
            title="single-model verification",
            subtitle=f"score={score:.6f} threshold={self.threshold:.2f} verdict={verdict}",
        )
        return VerifyArtifacts(
            score=score,
            pred=pred,
            verdict=verdict,
            reason=reason,
            left_yolo_conf=left_yolo_conf,
            right_yolo_conf=right_yolo_conf,
            left_qr_text=left_qr_text,
            right_qr_text=right_qr_text,
            left_overlay_b64=encode_jpg_base64(left_overlay),
            right_overlay_b64=encode_jpg_base64(right_overlay),
            left_warped_b64=encode_jpg_base64(left_warped),
            right_warped_b64=encode_jpg_base64(right_warped),
            left_mask_b64=encode_jpg_base64(binary_mask_to_bgr(left_mask > 127)),
            right_mask_b64=encode_jpg_base64(binary_mask_to_bgr(right_mask > 127)),
            panel_b64=encode_jpg_base64(panel),
            threshold=self.threshold,
        )

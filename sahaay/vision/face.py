"""Face tracking on the Hexagon NPU: BlazeFace detector + 468-point face mesh.

Models come from Qualcomm AI Hub (``mediapipe_face``, float, ONNX). Pre/post-processing is a
numpy re-implementation of the reference MediaPipe pipeline used by qai_hub_models
(letterbox -> detector -> anchor decode -> NMS -> rotated ROI -> affine crop -> landmarks -> map back).
No OpenCV: Windows on Snapdragon has no ARM64 wheel, so Pillow does the resampling.

Outputs per frame: 468 landmarks in frame pixels, a detection score, head pose (yaw/pitch/roll
as normalised ratios), eye aspect ratios (blink) and mouth opening (drag toggle).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..npu import SessionInfo, create_session

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# BlazeFace (back model) constants, from qai_hub_models.models.mediapipe_face.model
DETECTOR_SIZE = 256
LANDMARK_SIZE = 192
SCORE_CLIP = 100.0
BOX_SCALE = 1.1  # DETECT_DSCALE; DETECT_DXY is 0 so no directional offset
KP_RIGHT_EYE, KP_LEFT_EYE, KP_NOSE, KP_MOUTH = 1, 0, 2, 3

# Face mesh indices (MediaPipe canonical topology)
LM_NOSE_TIP = 1
LM_CHIN = 152
LM_FOREHEAD = 10
LM_LEFT_CHEEK, LM_RIGHT_CHEEK = 234, 454  # image-left / image-right
LM_LEFT_EYE_OUTER, LM_RIGHT_EYE_OUTER = 33, 263
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)
LM_LIP_TOP, LM_LIP_BOTTOM, LM_MOUTH_L, LM_MOUTH_R = 13, 14, 78, 308


@dataclass
class FaceResult:
    found: bool
    score: float = 0.0
    landmarks: np.ndarray | None = None  # [468, 3] x,y in frame pixels, z relative
    box: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 frame pixels
    roi: np.ndarray | None = None  # [4, 2] rotated ROI corners
    yaw: float = 0.0  # + when the nose points to image-right
    pitch: float = 0.0  # + when the nose points down
    roll: float = 0.0  # radians, eye line angle
    ear_left: float = 0.0
    ear_right: float = 0.0
    mouth_open: float = 0.0
    detector_ms: float = 0.0
    landmark_ms: float = 0.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _letterbox(img: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect, pad to square. Returns NCHW float [0,1], scale, (pad_x, pad_y)."""
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (px, py))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], scale, (px, py)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou <= iou_thr]
    return keep


def _affine_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 affine A such that A @ [x, y, 1] maps src points to dst points (3 point pairs)."""
    S = np.hstack([src, np.ones((3, 1))])  # 3x3
    # Solve S @ M = dst for M (3x2), then A = M.T (2x3)
    M = np.linalg.solve(S, dst)
    return M.T


class FaceTracker:
    """Runs the detector and the landmark model on every frame (both are sub-millisecond on the NPU)."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        prefer_npu: bool = True,
        min_box_score: float = 0.6,
        nms_iou: float = 0.3,
        min_landmark_score: float = 0.4,
    ) -> None:
        model_dir = Path(model_dir)
        self.det, self.det_info = create_session(model_dir / "face_detector.onnx", prefer_npu=prefer_npu)
        self.lmk, self.lmk_info = create_session(model_dir / "face_landmark_detector.onnx", prefer_npu=prefer_npu)
        anchors = np.load(ASSETS / "anchors_face_back.npy").astype(np.float32)
        self.anchors = anchors.reshape(anchors.shape[0], 2, 2)  # [[x_off, y_off], [x_scale, y_scale]]
        self.min_box_score = min_box_score
        self.nms_iou = nms_iou
        self.min_landmark_score = min_landmark_score

    # ---- public -----------------------------------------------------------------------
    @property
    def on_npu(self) -> bool:
        return self.det_info.on_npu and self.lmk_info.on_npu

    def infos(self) -> tuple[SessionInfo, SessionInfo]:
        return self.det_info, self.lmk_info

    def process(self, frame_rgb: np.ndarray) -> FaceResult:
        """frame_rgb: HxWx3 uint8 RGB."""
        img = Image.fromarray(frame_rgb)
        W, H = img.size
        t0 = time.perf_counter()
        det = self._detect(img)
        t1 = time.perf_counter()
        if det is None:
            return FaceResult(found=False, detector_ms=(t1 - t0) * 1000)
        box, kps, score = det
        roi = self._roi_from_detection(box, kps)
        landmarks, lscore = self._landmarks(img, roi)
        t2 = time.perf_counter()
        if landmarks is None or lscore < self.min_landmark_score:
            return FaceResult(found=False, score=float(score), box=box, roi=roi,
                              detector_ms=(t1 - t0) * 1000, landmark_ms=(t2 - t1) * 1000)
        res = FaceResult(found=True, score=float(lscore), landmarks=landmarks, box=box, roi=roi,
                         detector_ms=(t1 - t0) * 1000, landmark_ms=(t2 - t1) * 1000)
        self._pose_and_gestures(res)
        return res

    # ---- detector ---------------------------------------------------------------------
    def _detect(self, img: Image.Image):
        inp, scale, (px, py) = _letterbox(img, DETECTOR_SIZE)
        c1, c2, s1, s2 = self.det.run(None, {"image": inp})
        coords = np.concatenate([c1, c2], axis=1)[0]  # [896, 16]
        scores = np.concatenate([s1, s2], axis=1)[0, :, 0]  # [896]
        scores = _sigmoid(np.clip(scores, -SCORE_CLIP, SCORE_CLIP))
        cand = np.nonzero(scores >= self.min_box_score)[0]
        if cand.size == 0:
            return None
        coords = coords[cand].reshape(-1, 8, 2)
        anchors = self.anchors[cand]
        offset = anchors[:, 0:1, :] * DETECTOR_SIZE  # [n,1,2]
        scl = anchors[:, 1:2, :]
        mask = (np.arange(8) != 1).astype(np.float32)[:, None]  # wh gets no offset
        dec = coords * scl + offset * mask  # [n,8,2] in detector pixel space
        centers, wh, kps = dec[:, 0], dec[:, 1], dec[:, 2:]
        boxes = np.concatenate([centers - wh / 2, centers + wh / 2], axis=1)  # [n,4]
        keep = _nms(boxes, scores[cand], self.nms_iou)
        i = keep[0]  # highest score after NMS
        # detector space -> original frame space
        def back(p):
            return (p - np.array([px, py], dtype=np.float32)) / scale
        box = np.concatenate([back(boxes[i, :2]), back(boxes[i, 2:])])
        kp = back(kps[i])  # [6,2]
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3])), kp, float(scores[cand][i])

    def _roi_from_detection(self, box, kps: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = box
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0) * BOX_SCALE, (y1 - y0) * BOX_SCALE
        start, end = kps[KP_RIGHT_EYE], kps[KP_LEFT_EYE]
        theta = math.atan2(start[1] - end[1], start[0] - end[0])
        # unit square corners in the order (TL, BL, TR, BR), matching the reference
        pts = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32) * np.array([w / 2, h / 2], dtype=np.float32)
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
        return pts @ R.T + np.array([xc, yc], dtype=np.float32)

    # ---- landmarks --------------------------------------------------------------------
    def _landmarks(self, img: Image.Image, roi: np.ndarray):
        S = LANDMARK_SIZE
        dst = np.array([[0, 0], [0, S - 1], [S - 1, 0]], dtype=np.float64)
        A = _affine_from_points(roi[:3].astype(np.float64), dst)  # roi -> crop
        A3 = np.vstack([A, [0, 0, 1]])
        inv = np.linalg.inv(A3)[:2]  # crop -> roi (what PIL wants: output -> input mapping)
        crop = img.transform((S, S), Image.AFFINE, tuple(inv.flatten()), resample=Image.BILINEAR)
        inp = (np.asarray(crop, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
        score, lms = self.lmk.run(None, {"image": inp})
        lms = lms[0].astype(np.float32)  # [468,3] in [0,1] of the crop
        xy = lms[:, :2] * S
        ones = np.ones((xy.shape[0], 1), dtype=np.float32)
        xy_frame = np.hstack([xy, ones]) @ inv.T.astype(np.float32)  # back to frame pixels
        out = np.hstack([xy_frame, lms[:, 2:3] * S])
        return out, float(np.asarray(score).reshape(-1)[0])

    # ---- geometry ---------------------------------------------------------------------
    @staticmethod
    def _ear(lm: np.ndarray, idx) -> float:
        p = [lm[i, :2] for i in idx]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        h = np.linalg.norm(p[0] - p[3])
        return float((v1 + v2) / max(2.0 * h, 1e-6))

    def _pose_and_gestures(self, r: FaceResult) -> None:
        lm = r.landmarks
        assert lm is not None
        le, re = lm[LM_LEFT_EYE_OUTER, :2], lm[LM_RIGHT_EYE_OUTER, :2]
        r.roll = math.atan2(re[1] - le[1], re[0] - le[0])
        lc, rc = lm[LM_LEFT_CHEEK, :2], lm[LM_RIGHT_CHEEK, :2]
        top, chin = lm[LM_FOREHEAD, :2], lm[LM_CHIN, :2]
        nose = lm[LM_NOSE_TIP, :2]
        face_w = max(np.linalg.norm(rc - lc), 1e-6)
        face_h = max(np.linalg.norm(chin - top), 1e-6)
        # project the nose onto the roll-corrected face axes so yaw/pitch are independent of tilt
        c, s = math.cos(-r.roll), math.sin(-r.roll)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        centre = (lc + rc) / 2
        d = R @ (nose - centre)
        r.yaw = float(d[0] / (face_w / 2))
        mid_y = R @ ((top + chin) / 2 - centre)
        r.pitch = float((d[1] - mid_y[1]) / (face_h / 2))
        r.ear_left = self._ear(lm, LEFT_EYE_EAR)
        r.ear_right = self._ear(lm, RIGHT_EYE_EAR)
        lip = np.linalg.norm(lm[LM_LIP_TOP, :2] - lm[LM_LIP_BOTTOM, :2])
        mouth_w = max(np.linalg.norm(lm[LM_MOUTH_L, :2] - lm[LM_MOUTH_R, :2]), 1e-6)
        r.mouth_open = float(lip / mouth_w)

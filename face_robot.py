# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import cv2
import mediapipe as mp


_MIDLINE_TOP = 10      # top-center of forehead (on the face midline)
_MIDLINE_BOTTOM = 152  # bottom of chin (on the face midline)
_IRIS_A = 468          # one iris center
_IRIS_B = 473          # other iris center

_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "face_landmarker.task")
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")


class HalfRobotFilter:
    def __init__(self, robot_side='right', feather=15, max_num_faces=3, model_path=None):
        model_path = model_path or _DEFAULT_MODEL
        self.robot_side = robot_side
        self.feather = feather
        self._last_ts = -1

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_num_faces,   # <-- detect up to this many faces
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    @staticmethod
    def _landmarks_px(face_landmarks, w, h):
        n = len(face_landmarks)
        pts = np.empty((n, 2), dtype=np.int32)
        for i, lm in enumerate(face_landmarks):
            pts[i, 0] = int(lm.x * w)
            pts[i, 1] = int(lm.y * h)
        return pts

    def process(self, rgb_frame):
        h, w = rgb_frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb_frame))
        ts = max(self._last_ts + 1, int(time.monotonic() * 1000))
        self._last_ts = ts

        result = self.landmarker.detect_for_video(mp_image, ts)
        if not result.face_landmarks:
            return rgb_frame

        # Accumulate every face's effect onto one output frame.
        out = rgb_frame.copy()
        for face_landmarks in result.face_landmarks:
            pts = self._landmarks_px(face_landmarks, w, h)
            self._apply_to_face(out, pts)
        return out

    def _apply_to_face(self, frame, pts):
        h, w = frame.shape[:2]

        x, y, bw, bh = cv2.boundingRect(pts)
        pad = int(0.10 * max(bw, bh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        if x1 <= x0 or y1 <= y0:
            return

        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]

        lpts = pts.copy()
        lpts[:, 0] -= x0
        lpts[:, 1] -= y0

        hull = cv2.convexHull(lpts)
        face_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull, 255)

        ax, ay = lpts[_MIDLINE_TOP]
        bx, by = lpts[_MIDLINE_BOTTOM]
        ys, xs = np.mgrid[0:rh, 0:rw]
        cross = (bx - ax) * (ys - ay) - (by - ay) * (xs - ax)
        side = (cross > 0) if self.robot_side == 'right' else (cross < 0)
        side_mask = np.where(side, 255, 0).astype(np.uint8)

        robot_mask = cv2.bitwise_and(face_mask, side_mask)

        robot_roi = self._robotize(roi)
        alpha = cv2.GaussianBlur(robot_mask, (0, 0), self.feather).astype(np.float32) / 255.0
        alpha = alpha[..., None]
        blended = (roi.astype(np.float32) * (1.0 - alpha) +
                   robot_roi.astype(np.float32) * alpha).astype(np.uint8)

        frame[y0:y1, x0:x1] = blended
        self._draw_robot_eye(frame, pts)

    @staticmethod
    def _robotize(rgb_roi):
        gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        steel = np.empty((*gray.shape, 3), dtype=np.uint8)
        steel[..., 0] = (gray * 0.60).astype(np.uint8)
        steel[..., 1] = (gray * 0.80).astype(np.uint8)
        steel[..., 2] = np.clip(gray.astype(np.int32) + 30, 0, 255).astype(np.uint8)

        edges = cv2.Canny(gray, 80, 160)
        steel[edges > 0] = (0, 255, 255)
        return steel

    def _draw_robot_eye(self, frame, pts):
        if pts.shape[0] < 478:
            return
        ax, ay = pts[_MIDLINE_TOP]
        bx, by = pts[_MIDLINE_BOTTOM]
        for idx in (_IRIS_A, _IRIS_B):
            ex, ey = pts[idx]
            cross = (bx - ax) * (ey - ay) - (by - ay) * (ex - ax)
            on_robot = (cross > 0) if self.robot_side == 'right' else (cross < 0)
            if on_robot:
                cv2.circle(frame, (int(ex), int(ey)), 14, (0, 255, 255), 2)
                cv2.circle(frame, (int(ex), int(ey)), 6, (255, 80, 0), -1)

    def close(self):
        self.landmarker.close()
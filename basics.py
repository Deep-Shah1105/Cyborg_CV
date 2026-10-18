# -*- coding: utf-8 -*-
import numpy as np
import cv2  # conda install opencv
from numba import njit  # conda install numba


@njit(cache=True)
def histogram_figure_numba(np_img):
    h = np_img.shape[0]
    w = np_img.shape[1]

    r_bars = np.zeros(256, dtype=np.float64)
    g_bars = np.zeros(256, dtype=np.float64)
    b_bars = np.zeros(256, dtype=np.float64)

    for y in range(h):
        for x in range(w):
            r_bars[np_img[y, x, 0]] += 1.0
            g_bars[np_img[y, x, 1]] += 1.0
            b_bars[np_img[y, x, 2]] += 1.0

    # Normalize to [0, 3] using the global peak so no bar ever exceeds the y-limit.
    max_val = 0.0
    for i in range(256):
        if r_bars[i] > max_val:
            max_val = r_bars[i]
        if g_bars[i] > max_val:
            max_val = g_bars[i]
        if b_bars[i] > max_val:
            max_val = b_bars[i]

    if max_val > 0.0:
        scale = 3.0 / max_val
        for i in range(256):
            r_bars[i] *= scale
            g_bars[i] *= scale
            b_bars[i] *= scale

    return r_bars, g_bars, b_bars


def image_statistics(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    counts = np.bincount(gray.ravel(), minlength=256)
    return [
        f"Mean: {float(np.mean(gray)):6.2f}",
        f"Mode: {int(np.argmax(counts)):6d}",
        f"Std:  {float(np.std(gray)):6.2f}",
        f"Min:  {int(np.min(gray)):6d}",
        f"Max:  {int(np.max(gray)):6d}",
    ]


def image_entropy(np_img):

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    counts = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def linear_transformation(np_img, alpha=1.3, beta=15):
    return cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)


def histogram_equalization(np_img):
    r, g, b = cv2.split(np_img)
    return cv2.merge([cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)])


def apply_filter(np_img, kind):
    if kind == 'blur':
        return cv2.GaussianBlur(np_img, (7, 7), 0)

    if kind == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(np_img, -1, kernel)

    if kind == 'sobel':
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

    if kind == 'canny':
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    if kind == 'gabor':
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC1, kernel)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

    # unknown kind -> no change (caller normally guards against this)
    return np_img

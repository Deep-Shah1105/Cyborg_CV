# -*- coding: utf-8 -*-
""""
Keyboard toggles (debounced to avoid double-fire at high fps):
  h  histogram overlay on/off
  t  text/HUD on/off
  r  half-robot effect on/off
  e  histogram equalization on/off
  l  linear transformation on/off
  f  cycle filter: none -> sobel -> canny -> blur -> sharpen -> gabor
  q  quit (handled inside capturing.py)
"""

import numpy as np
import cv2
import keyboard  

from capturing import VirtualCamera
from overlays import (initialize_hist_figure, plot_overlay_to_image,
                      plot_strings_to_image, update_histogram)
from basics import (histogram_figure_numba, image_statistics, image_entropy,
                    linear_transformation, histogram_equalization, apply_filter)
from face_robot import HalfRobotFilter


class Toggles:
    def __init__(self, cooldown=8):
        self.cooldown = cooldown
        self._cool = {}
        self.state = {
            'hist': True,
            'text': True,
            'robot': True,
            'equalize': False,
            'linear': False,
        }
        self.filters = ['none', 'sobel', 'canny', 'blur', 'sharpen', 'gabor']
        self.filter_idx = 0

    def update(self):
        # decay all cooldowns
        for k in list(self._cool):
            if self._cool[k] > 0:
                self._cool[k] -= 1

        keymap = {'h': 'hist', 't': 'text', 'r': 'robot', 'e': 'equalize', 'l': 'linear'}
        for key, name in keymap.items():
            if keyboard.is_pressed(key) and self._cool.get(key, 0) == 0:
                self.state[name] = not self.state[name]
                self._cool[key] = self.cooldown

        if keyboard.is_pressed('f') and self._cool.get('f', 0) == 0:
            self.filter_idx = (self.filter_idx + 1) % len(self.filters)
            self._cool['f'] = self.cooldown

    @property
    def current_filter(self):
        return self.filters[self.filter_idx]


def custom_processing(img_source_generator, width, height):
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    toggles = Toggles()


    robot = HalfRobotFilter(robot_side='right', feather=15, max_num_faces=3)
    robot_ok = True

    for sequence in img_source_generator:
        # Capture yields a reversed view (bgr_to_rgb=True). Make it a contiguous,
        # writeable RGB uint8 array so numba/cv2/mediapipe behave predictably.
        sequence = np.ascontiguousarray(sequence)

        toggles.update()

        # --- Basics: point operations (toggleable) ---
        if toggles.state['linear']:
            sequence = linear_transformation(sequence, alpha=1.3, beta=15)
        if toggles.state['equalize']:
            sequence = histogram_equalization(sequence)

        # --- Basics: chosen filter (cycled with 'f') ---
        if toggles.current_filter != 'none':
            sequence = apply_filter(sequence, toggles.current_filter)

        # --- Special task: Half Human / Half Robot (neural network) ---
        if toggles.state['robot'] and robot_ok:
            try:
                sequence = np.ascontiguousarray(robot.process(sequence))
            except Exception as e:
                print(f"[warn] robot.process failed on a frame: {e}")

        # --- Histogram overlay ---
        if toggles.state['hist']:
            r_bars, g_bars, b_bars = histogram_figure_numba(sequence)
            update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
            sequence = plot_overlay_to_image(sequence, fig)

        # --- Text HUD ---
        if toggles.state['text']:
            lines = image_statistics(sequence)
            lines.append(f"Entropy: {image_entropy(sequence):4.2f}")
            lines.append(f"Filter: {toggles.current_filter}")
            sequence = plot_strings_to_image(sequence, lines)

        if sequence.shape[1] != width or sequence.shape[0] != height:
            sequence = cv2.resize(sequence, (width, height))

        yield np.ascontiguousarray(sequence)


def main():
    width = 1280
    height = 720
    fps = 30

    vc = VirtualCamera(fps, width, height)
    vc.virtual_cam_interaction(
        custom_processing(
            vc.capture_cv_video(2, bgr_to_rgb=True),
            width, height,
        )
    )


if __name__ == "__main__":
    main()
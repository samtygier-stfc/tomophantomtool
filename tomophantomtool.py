#!/usr/bin/env python3
# Copyright (C) 2024 ISIS Rutherford Appleton Laboratory UKRI
# SPDX - License - Identifier: GPL-3.0-or-later

from datetime import datetime
import sys
from math import sqrt
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tomophantom
import yaml
from PIL import Image, ImageDraw
from tifffile import tifffile

path_library3D = Path(tomophantom.__file__).parent / "phantomlib" / "Phantom3DLibrary.dat"

class TomoPhantomTool:
    def __init__(self, input_file: Path, output_dir: Path):
        self.input_file = input_file
        self.output_dir = output_dir

        self.settings = yaml.safe_load(input_file.open())

        print(self.settings)
        self.create_model()
        self.debug_labels = self.settings["debug_labels"]
        # self.stacks = {}

    def create_model(self):
        print(path_library3D)
        self.size = [self.settings["n_size"], self.settings["n_size"]]

        if "linspace" in self.settings["angles"]:
            self.angles = np.linspace(*self.settings["angles"]["linspace"],
                                      dtype='float32')
        elif "golden" in self.settings["angles"]:
            start, stop, count = self.settings["angles"]["golden"]
            assert start == 0
            golden_angle = 180 * (3 - sqrt(5))
            self.angles = np.array([(n * golden_angle) % stop for n in range(count)],
                                   dtype='float32')
        else:
            raise ValueError(
                f"Unknown angle specification: {self.settings['angles']}")

    def create_stacks(self):
        for stack_type, stack in self.settings["stacks"].items():
            if stack_type == "sample":
                self.create_sample(stack_type, stack)
            elif stack_type == "180_deg":
                self.create_sample(stack_type, stack, 180)
            else:
                if "flat" in stack_type:
                    value = 1
                else:
                    value = 0
                self.create_stack(stack_type, stack, value)

    @staticmethod
    def make_label(size, label):
        img = Image.new("L", size)
        draw = ImageDraw.Draw(img)
        draw.text([size[0] // 5, size[1] // 5], label, fill=255)
        return np.array(img) / 256

    def create_sample(self, name, sample_settings, angle: float | None = None):
        if angle is None:
            angles = self.angles
        else:
            angles = np.array([angle], dtype=np.float32)

        sample_absorb = tomophantom.TomoP3D.ModelSino(self.settings["model"], int(self.size[1] / sqrt(2)),
                                          self.size[0], self.size[1], angles,
                                          str(path_library3D))

        coef = 2.0 / self.size[0]  # to give a good contrast range
        sample_transmission = np.exp(-coef * sample_absorb)
        print("Sample: absorb", sample_absorb.min(), sample_absorb.max())
        print("Sample: trans ", sample_transmission.min(),
              sample_transmission.max())
        print("Sample shape:", sample_transmission.shape)

        if self.debug_labels:
            for n, a in enumerate(angles):
                text = self.make_label(self.size, f"{name} [{n}]:{a:.2f}deg")
                sample_transmission[:, n, :] -= text
            sample_transmission = sample_transmission.clip(0, 1)

        self.apply_effects(sample_transmission)
        self.save_images(sample_transmission, sample_settings["subdir"],
                         sample_settings["pattern"], angles)

        if "imat_log" in sample_settings:
            self.save_imat_log(angles, sample_settings["subdir"], sample_settings["imat_log"])


    def create_stack(self, name, stack_settings, value):
        shape = [self.size[1], stack_settings["count"], self.size[0]]
        stack_transmission = np.ones(shape, dtype=np.float32) * value

        mean_value = stack_transmission[:, 0, :].mean()
        if self.debug_labels:
            for n in range(stack_settings["count"]):
                text = self.make_label(self.size, f"{name} [{n}]")
                if mean_value > 0.5:
                    stack_transmission[:, n, :] -= text
                else:
                    stack_transmission[:, n, :] += text
            stack_transmission = stack_transmission.clip(0, 1)

        self.apply_effects(stack_transmission)
        self.save_images(stack_transmission, stack_settings["subdir"],
                         stack_settings["pattern"])

    def apply_effects(self, stack):
        for effect_name, effect_settings in self.settings["effects"].items():
            if effect_name == "dark":
                self.apply_effect_dark(stack, effect_settings)
            elif effect_name == "hot_pixels":
                self.apply_effect_hot_pixels(stack, effect_settings)
            elif effect_name == "shot_to_shot":
                self.apply_effect_shot_to_shot(stack, effect_settings)

    def apply_effect_dark(self, stack, settings):
        if "uniform" in settings:
            stack += float(settings["uniform"])
        else:
            raise NotImplementedError("Unknown dark settings")

    def apply_effect_hot_pixels(self, stack, settings):
        if hasattr(self, "hot_pixels_mask"):
            mask = self.hot_pixels_mask
        else:
            density = float(settings["density"])
            shape = [self.size[1], 1, self.size[0]]
            mask = np.random.uniform(size=shape) < density
            self.hot_pixels_mask = mask

        if "add" in settings:
            stack += float(settings["add"]) * mask
        else:
            raise NotImplementedError("Unknown dark settings")

    def apply_effect_shot_to_shot(self, stack, settings):
        if "random" in settings:
            variation = np.random.normal(1,float(settings["random"]), size=(1,stack.shape[1],1))
            stack *= variation
        else:
            raise NotImplementedError("Unknown shot_to_shot settings")

    def save_images(self, data: np.ndarray, subdir: str, pattern: str, angles: Optional[np.ndarray] = None):
        outscale = 1
        outmode = np.float32

        out_subdir = self.output_dir / subdir
        try:
            out_subdir.mkdir(parents=True)
        except FileExistsError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(0, data.shape[1]):
                values = {'projection': i}
                if angles is not None:
                    values['angle'] = angles[i]
                file_path = out_subdir / pattern.format(i, **values)
                tifffile.imwrite(file_path,
                            (data[:, i, :] * outscale).astype(outmode),
                            compression="ZLIB")

    def save_imat_log(self, angles: np.ndarray, subdir: str, file_name: str):
        log_header = "TIME STAMP,IMAGE TYPE,IMAGE COUNTER,COUNTS BM3 before image,COUNTS BM3 after image\n"
        log_template = "{date},Projection,{projection},Angle:{angle},Monitor 3 before: 73769,Monitor 3 after: 203948\n"

        file_path = self.output_dir / file_name
        date_s = datetime.utcnow().ctime()
        with open(file_path, "w") as log_file:
            log_file.write(log_header)
            for n, angle in enumerate(angles):
                line = log_template.format(date=date_s, projection=n, angle=angle)
                log_file.write(line)


if __name__ == "__main__":
    input_file = Path(sys.argv[1])

    output_path = Path(sys.argv[2])
    if output_path.exists():
        print(f"Output path exists: {sys.argv[2]}")
        sys.exit(1)

    tpt = TomoPhantomTool(input_file, output_path)
    tpt.create_stacks()

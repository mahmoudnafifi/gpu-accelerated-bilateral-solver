"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
Mahmoud Afifi (m.afifi1@samsung.com, m.3afifi@gmail.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

This demo illustrates the usage of the GPU-accelerated bilateral solver.
"""


import os
import cv2
import numpy as np
import argparse
import torch

from gpu_solver import gpu_bilateral_solver


def imread(img_file: str, single_channel: bool = False) -> np.ndarray:
    """Reads any 8-bit or 16-bit image and returns a normalized float32 image in [0,1]."""
    if not os.path.exists(img_file):
      raise FileNotFoundError(f'Image not found: {img_file}')
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img is None:
      raise FileNotFoundError(f'Cannot load image: {img_file}')
    if img.ndim == 3 and img.shape[2] == 3 and not single_channel:
      img = img[..., ::-1]
    if img.dtype == np.uint8:
      img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
      img = img.astype(np.float32) / 65535.0
    else:
      img = img.astype(np.float32)
    if img.ndim == 2:
      img = img[..., None]
    return img

def get_args():
    parser = argparse.ArgumentParser(description='GPU-accelerated bilateral solver demo')
    parser.add_argument('--input-image-path', type=str, required=True, help='Path to input image')
    parser.add_argument('--reference-image-path', type=str, required=True, help='Path to reference image')
    parser.add_argument('--iterations', type=int, default=80, help='Number of SOR iterations for the bilateral solver.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    in_img = imread(args.input_image_path)
    ref_img = imread(args.reference_image_path)
    if in_img.shape[:2] != ref_img.shape[:2]:
        print('Warning: Input image and reference image have different sizes. Resizing input to match reference.')
        h, w = ref_img.shape[:2]
        in_img = cv2.resize(in_img, (w, h), interpolation=cv2.INTER_AREA)
    in_tensor = torch.from_numpy(in_img).permute(2, 0, 1).unsqueeze(0)
    ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Warning: GPU not detected. This solver is much slower on CPU.')
        print('   Recommended: run on a machine with a GPU.')
    in_tensor = in_tensor.to(device)
    ref_tensor = ref_tensor.to(device)

    refined = gpu_bilateral_solver(guide=ref_tensor, coeff_map=in_tensor, n_iter=args.iterations)
    refined_img = refined.squeeze(0).permute(1, 2, 0).cpu().numpy()
    refined_img = np.clip(refined_img, 0.0, 1.0)
    root, ext = os.path.splitext(args.input_image_path)
    out_path = root + '_output.png'
    cv2.imwrite(out_path, (refined_img[..., ::-1] * 255).astype(np.uint8))

    print(f'Saved refined image to: {out_path}')

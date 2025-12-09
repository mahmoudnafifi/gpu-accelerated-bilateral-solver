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

Implementation of the GPU-accelerated bilateral solver, as described in the paper:

“Modular Neural Image Signal Processing.”

"""


import torch
import torch.nn.functional as F
from typing import Optional

EPS = 0.00000001

def gpu_bilateral_solver(guide: torch.Tensor, coeff_map: torch.Tensor,
                      k: Optional[int] = 7, sigma_spatial: Optional[float] = 3.0, sigma_luma: Optional[float] = 0.01,
                      lam: Optional[float] = 1e-3, n_iter: Optional[int] = 80,
                      omega: Optional[float] = 1.6):
  """GPU-accelerated iterative bilateral solver for edge-aware refinement. This function refines an input coefficient
     map using a guidance image by minimizing a bilateral-smoothing quadratic objective.
     (Reference paper: "Modular Neural Image Signal Processing").
  Args:
    guide (Tensor): Guidance tensor of shape (B, 1 or 3, H, W). If RGB, luminance is computed internally using fixed
       weights (0.2989, 0.5870, 0.1140).
    coeff_map (Tensor): Input tensor to be refined, of shape (B, C, H, W).
    k (int, optional): Neighborhood size (kernel size). Must be odd.
    sigma_spatial (float, optional): Spatial kernel standard deviation (pixels). Controls spatial decay of bilateral
       affinities.
    sigma_luma (float, optional): Range kernel standard deviation for luminance differences. Controls edge sensitivity.
    lam (float, optional): Smoothness vs. data-fidelity trade-off λ. Higher values preserve the input more strongly.
    n_iter (int, optional): Number of SOR iterations.
    omega (float, optional): SOR relaxation parameter in [1, 2). Values > 1 accelerate convergence.
  Returns:
      Tensor: Refined tensor of shape (B, C, H, W), with improved spatial coherence and edge preservation.
  """
  if guide.shape[1] == 3:
    guide = 0.2989 * guide[:, 0:1] + 0.5870 * guide[:, 1:2] + 0.1140 * guide[:, 2:3]

  b, c, h, w = coeff_map.shape
  pad = k // 2
  device = coeff_map.device
  dtype = coeff_map.dtype

  # pre-computed bilateral weights (fixed for all iterations).
  guide_p = F.pad(guide, (pad, pad, pad, pad), mode='reflect')
  neigh_guide = F.unfold(guide_p, kernel_size=k, padding=0).view(b, 1, k * k, h, w)
  center_guide = guide.unsqueeze(2)
  diff2 = (neigh_guide - center_guide).pow(2)
  range_w = torch.exp(-diff2 / (2 * (sigma_luma ** 2)))
  yy, xx = torch.meshgrid(
    torch.arange(-pad, pad + 1, device=device, dtype=dtype),
    torch.arange(-pad, pad + 1, device=device, dtype=dtype),
    indexing="ij"
  )
  spatial = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (sigma_spatial ** 2)))
  spatial = spatial.reshape(1, 1, k * k, 1, 1)
  w_b = range_w * spatial
  w_b = w_b / w_b.sum(dim=2, keepdim=True).clamp_min(EPS)
  out = coeff_map.clone()
  inv_alpha = 1.0 / (lam + 1.0)

  for _ in range(n_iter):
    out_p = F.pad(out, (pad, pad, pad, pad), mode='reflect')
    neigh_coeff = F.unfold(out_p, kernel_size=k, padding=0).view(b, c, k * k, h, w)
    smooth = (neigh_coeff * w_b).sum(dim=2)
    target = (lam * coeff_map + smooth) * inv_alpha
    out = out + omega * (target - out)
  return out
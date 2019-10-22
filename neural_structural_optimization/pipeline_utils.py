# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pipeline utilties."""
import math
from typing import Any, Dict

import matplotlib.cm
import matplotlib.colors
from neural_structural_optimization import problems
import numpy as np
from PIL import Image
import xarray


def image_from_array(
    data: np.ndarray, cmap: str = 'Greys', vmin: float = 0, vmax: float = 1,
) -> Image.Image:
  """Convert a NumPy array into a Pillow Image using a colormap."""
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
  frame = np.ma.masked_invalid(data)
  image = Image.fromarray(mappable.to_rgba(frame, bytes=True), mode='RGBA')
  return image


def image_from_design(
    design: xarray.DataArray, problem: problems.Problem,
) -> Image.Image:
  """Convert a design and problem into a Pillow Image."""
  assert design.dims == ('y', 'x'), design.dims
  imaged_designs = []
  if problem.mirror_left:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  imaged_designs.append(design)
  if problem.mirror_right:
    imaged_designs.append(design.isel(x=slice(None, None, -1)))
  return image_from_array(xarray.concat(imaged_designs, dim='x').data)


def dynamic_depth_kwargs(problem: problems.Problem) -> Dict[str, Any]:
  max_resize = min(math.gcd(problem.width, problem.height),
                   round(math.sqrt(problem.width * problem.height) / 4))
  resizes = [1] + [2] * int(math.log2(max_resize)) + [1]
  conv_filters = [512, 256, 128, 64, 32, 16, 8, 1][-len(resizes):]
  assert len(conv_filters) == len(resizes)
  return dict(
      resizes=resizes,
      conv_filters=conv_filters,
      dense_channels=conv_filters[0] // 2,
  )

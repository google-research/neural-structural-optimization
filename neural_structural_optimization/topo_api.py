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

# pylint: disable=missing-docstring

import autograd.numpy as np
from neural_structural_optimization import topo_physics


def specified_task(problem):
  """Given a problem, return parameters for running a topology optimization."""
  fixdofs = np.flatnonzero(problem.normals.ravel())
  alldofs = np.arange(2 * (problem.width + 1) * (problem.height + 1))
  freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

  params = {
      # material properties
      'young': 1,
      'young_min': 1e-9,
      'poisson': 0.3,
      'g': 0,
      # constraints
      'volfrac': problem.density,
      'xmin': 0.001,
      'xmax': 1.0,
      # input parameters
      'nelx': problem.width,
      'nely': problem.height,
      'mask': problem.mask,
      'freedofs': freedofs,
      'fixdofs': fixdofs,
      'forces': problem.forces.ravel(),
      'penal': 3.0,
      'filter_width': 2,
  }
  return params


class Environment:

  def __init__(self, args):
    self.args = args
    self.ke = topo_physics.get_stiffness_matrix(
        self.args['young'], self.args['poisson'])

  def reshape(self, params):
    return params.reshape(self.args['nely'], self.args['nelx'])

  def render(self, params, volume_contraint=True):
    return topo_physics.physical_density(
        self.reshape(params), self.args, volume_contraint=volume_contraint,
    )

  def objective(self, params, volume_contraint=False):
    return topo_physics.objective(
        self.reshape(params), self.ke, self.args,
        volume_contraint=volume_contraint,
    )

  def constraint(self, params):
    volume = topo_physics.mean_density(self.reshape(params), self.args)
    return volume - self.args['volfrac']

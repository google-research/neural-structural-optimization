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
# pylint: disable=invalid-name
# pylint: disable=unused-variable

import os
import autograd
import autograd.numpy as np
from autograd.test_util import check_grads
from neural_structural_optimization import topo_physics
import numpy as npo
from absl.testing import absltest


######### HELPER FUNCTIONS #########
def get_mini_problem():
  args = topo_physics.default_args()
  args['nely'], args['nelx'] = 10, 15

  left_wall = list(range(0, 2*(args['nely']+1), 2))
  right_corner = [2*(args['nelx']+1)*(args['nely']+1)-1]
  fixdofs = np.asarray(left_wall + right_corner)
  alldofs = np.arange(2*(args['nely']+1)*(args['nelx']+1))
  freedofs = np.asarray(list(set(alldofs) - set(fixdofs)))

  args['freedofs'], args['fixdofs'] = freedofs, fixdofs
  args['forces'] = np.zeros(2*(args['nely']+1)*(args['nelx']+1))
  args['forces'][1] = -1

  coeffs = np.ones((args['nely'], args['nelx'])) * args['volfrac']
  ke = topo_physics.get_stiffness_matrix(
      young=args['young'], poisson=args['poisson'])
  u = topo_physics.displace(
      coeffs, ke, args['forces'], args['freedofs'], args['fixdofs'])

  return args, coeffs, ke, u


def old_compliance_fn(x, u, ke, penal):  # differentiable
  #  Calculates the compliance. We replaced this (slow) loop version
  #  with a tensor-contrating version. It's useful to test that the old
  #  version and the new version of the compliance function are calculating
  #  the same quantity, even though they do it in different ways
  c = 0
  nely, nelx = x.shape
  for ely in range(nely):
    for elx in range(nelx):
      n1 = (nely+1)*(elx+0) + (ely+0)
      n2 = (nely+1)*(elx+1) + (ely+0)
      n3 = (nely+1)*(elx+1) + (ely+1)
      n4 = (nely+1)*(elx+0) + (ely+1)
      ixs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

      ue = u[ixs]
      c = c + x[ely, elx]**penal * np.matmul(np.matmul(ue.T, ke), ue)
  return c


######### PHYSICS TESTS #########
class TopoPhysicsTest(absltest.TestCase):

  def test_high_density(self):
    # try running simulation with volfrac close to 1
    # resulting structure should be ALMOST all the way filled in
    args, coeffs, ke, u = get_mini_problem()
    args['volfrac'] = 1

    l, x, frames = topo_physics.run_toposim(
        args=args, loss_only=False, verbose=False)
    npo.testing.assert_almost_equal(
        actual=topo_physics.mean_density(x, args), desired=1, decimal=4)

  def test_compliance_sign(self):
    # compliance gradients should ALL always be greater than 0
    args, coeffs, ke, u = get_mini_problem()
    c, dc = autograd.value_and_grad(topo_physics.compliance)(coeffs, u, ke)
    assert dc.min() > 0

  def test_compliance_numerics(self):
    # compare new (tensor-contracting) version against the old (loop) version
    args, coeffs, ke, u = get_mini_problem()
    coeffs = np.random.rand(*coeffs.shape) * 0.4

    c = topo_physics.compliance(coeffs, u, ke, penal=args['penal'])
    c_old = old_compliance_fn(coeffs, u, ke, args['penal'])
    npo.testing.assert_almost_equal(actual=c, desired=c_old, decimal=5)

  def test_sigmoid(self):
    x = np.random.randn(5)
    actual = topo_physics.logit(topo_physics.sigmoid(x))
    npo.testing.assert_almost_equal(actual, x, decimal=6)

  def test_structure(self):
    nelx, nely = 60, 20

    left_wall = list(range(0, 2*(nely+1), 2))
    right_corner = [2*(nelx+1)*(nely+1)-1]
    fixdofs = np.asarray(left_wall + right_corner)
    alldofs = np.arange(2*(nely+1)*(nelx+1))
    freedofs = np.array(list(set(alldofs) - set(fixdofs)))

    forces = np.zeros(2*(nely+1)*(nelx+1))
    forces[1] = -1.0

    args = topo_physics.default_args()
    args.update({'nelx': nelx,
                 'nely': nely,
                 'freedofs': freedofs,
                 'fixdofs': fixdofs,
                 'forces': forces})

    _, x, _ = topo_physics.run_toposim(
        args=args, loss_only=False, verbose=False)
    x = abs(x)  # remove negative zeros!

    path = os.path.join(os.path.dirname(__file__), 'truss_test.csv')
    # To regenerate, run the test binary directly, e.g., with
    # python ./neural_structural_optimization/topo_physics_test
    # after uncommenting this line:
    # np.savetxt(path, x, delimiter=",", fmt='%.0f',)
    target_struct = np.loadtxt(path, delimiter=',')

    npo.testing.assert_array_equal(x.round(0), target_struct)

  def test_displace_gradients(self):
    args, coeffs, ke, u = get_mini_problem()
    var_names = ['penal', 'forces', 'freedofs', 'fixdofs']
    [penal, forces, freedofs, fixdofs] = [args[k] for k in var_names]
    pos_args = (forces, freedofs, fixdofs)
    kwargs = dict(
        penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
    check_grads(
        lambda x: topo_physics.displace(x, ke, *pos_args, **kwargs),
        modes=['rev'])(coeffs)

  def test_toposim_gradients(self):
    # is the entire simulation differentiable?
    args, coeffs, ke, u = get_mini_problem()
    args['opt_steps'] = 3
    np.random.seed(0)
    try:
      original_rtol = autograd.test_util.RTOL
      autograd.test_util.RTOL = 1e-5
      check_grads(lambda x: topo_physics.run_toposim(x, args), modes=['rev'])(
          coeffs
      )
    finally:
      autograd.test_util.RTOL = original_rtol


if __name__ == '__main__':
  absltest.main()

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
"""Autograd implementation of topology optimization for compliance minimization.

Exactly reproduces the result of "Efficient topology optimization in MATLAB
using 88 lines of code":
http://www.topopt.mek.dtu.dk/Apps-and-software/Efficient-topology-optimization-in-MATLAB
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=superfluous-parens

import autograd
import autograd.numpy as np
from neural_structural_optimization import autograd_lib
from neural_structural_optimization import caching

# A note on conventions:
# - forces and freedofs are stored flattened, but logically represent arrays of
#   shape (Y+1, X+1, 2)
# - mask is either a scalar (1) or an array of shape (X, Y).
# Yes, this is confusing. Sorry!


def default_args():
  # select the degrees of freedom
  nely = 25
  nelx = 80

  left_wall = list(range(0, 2*(nely+1), 2))
  right_corner = [2*(nelx+1)*(nely+1)-1]
  fixdofs = np.asarray(left_wall + right_corner)
  alldofs = np.arange(2*(nely+1)*(nelx+1))
  freedofs = np.asarray(list(set(alldofs) - set(fixdofs)))

  forces = np.zeros(2*(nely+1)*(nelx+1))
  forces[1] = -1.0

  return {'young': 1,     # material properties
          'young_min': 1e-9,
          'poisson': 0.3,
          'g': 0,  # force of gravity
          'volfrac': 0.4,  # constraints
          'nelx': nelx,     # input parameters
          'nely': nely,
          'freedofs': freedofs,
          'fixdofs': fixdofs,
          'forces': forces,
          'mask': 1,
          'penal': 3.0,
          'rmin': 1.5,
          'opt_steps': 50,
          'filter_width': 2,
          'step_size': 0.5,
          'name': 'truss'}


def physical_density(x, args, volume_contraint=False, cone_filter=True):
  shape = (args['nely'], args['nelx'])
  assert x.shape == shape or x.ndim == 1
  x = x.reshape(shape)
  if volume_contraint:
    mask = np.broadcast_to(args['mask'], x.shape) > 0
    x_designed = sigmoid_with_constrained_mean(x[mask], args['volfrac'])
    x_flat = autograd_lib.scatter1d(
        x_designed, np.flatnonzero(mask), x.size)
    x = x_flat.reshape(x.shape)
  else:
    x = x * args['mask']
  if cone_filter:
    x = autograd_lib.cone_filter(x, args['filter_width'], args['mask'])
  return x


def mean_density(x, args, volume_contraint=False, cone_filter=True):
  return (np.mean(physical_density(x, args, volume_contraint, cone_filter))
          / np.mean(args['mask']))


def get_stiffness_matrix(young, poisson):
  # Element stiffness matrix
  e, nu = young, poisson
  k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
  return e/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
                              ])


@caching.ndarray_safe_lru_cache(1)
def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
  index_map = autograd_lib.inverse_permutation(
      np.concatenate([freedofs, fixdofs]))
  keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
  i = index_map[k_ylist][keep]
  j = index_map[k_xlist][keep]
  return index_map, keep, np.stack([i, j])


def displace(x_phys, ke, forces, freedofs, fixdofs, *,
             penal=3, e_min=1e-9, e_0=1):
  # Displaces the load x using finite element techniques. The spsolve here
  # occupies the majority of this entire simulation's runtime.
  stiffness = young_modulus(x_phys, e_0, e_min, p=penal)
  k_entries, k_ylist, k_xlist = get_k(stiffness, ke)

  index_map, keep, indices = _get_dof_indices(
      freedofs, fixdofs, k_ylist, k_xlist
  )
  u_nonzero = autograd_lib.solve_coo(k_entries[keep], indices, forces[freedofs],
                                     sym_pos=True)
  u_values = np.concatenate([u_nonzero, np.zeros(len(fixdofs))])

  return u_values[index_map]


def get_k(stiffness, ke):
  # Constructs a sparse stiffness matrix, k, for use in the displace function.
  nely, nelx = stiffness.shape

  # get position of the nodes of each element in the stiffness matrix
  ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords
  ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

  n1 = (nely+1)*(elx+0) + (ely+0)
  n2 = (nely+1)*(elx+1) + (ely+0)
  n3 = (nely+1)*(elx+1) + (ely+1)
  n4 = (nely+1)*(elx+0) + (ely+1)
  edof = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])
  edof = edof.T[0]

  x_list = np.repeat(edof, 8)  # flat list pointer of each node in an element
  y_list = np.tile(edof, 8).flatten()  # flat list pointer of each node in elem

  # make the stiffness matrix
  kd = stiffness.T.reshape(nelx*nely, 1, 1)
  value_list = (kd * np.tile(ke, kd.shape)).flatten()
  return value_list, y_list, x_list


def young_modulus(x, e_0, e_min, p=3):
  return e_min + x ** p * (e_0 - e_min)


def compliance(x_phys, u, ke, *, penal=3, e_min=1e-9, e_0=1):
  # Calculates the compliance
  # Read about how this was vectorized here:
  # https://colab.research.google.com/drive/1PE-otq5hAMMi_q9dC6DkRvf2xzVhWVQ4

  # index map
  nely, nelx = x_phys.shape
  ely, elx = np.meshgrid(range(nely), range(nelx))  # x, y coords

  # nodes
  n1 = (nely+1)*(elx+0) + (ely+0)
  n2 = (nely+1)*(elx+1) + (ely+0)
  n3 = (nely+1)*(elx+1) + (ely+1)
  n4 = (nely+1)*(elx+0) + (ely+1)
  all_ixs = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1])

  # select from u matrix
  u_selected = u[all_ixs]

  # compute x^penal * U.T @ ke @ U in a vectorized way
  ke_u = np.einsum('ij,jkl->ikl', ke, u_selected)
  ce = np.einsum('ijk,ijk->jk', u_selected, ke_u)
  C = young_modulus(x_phys, e_0, e_min, p=penal) * ce.T
  return np.sum(C)


def optimality_criteria_combine(x, dc, dv, args, max_move=0.2, eta=0.5):
  """Fully differentiable version of the optimality criteria."""

  volfrac = args['volfrac']

  def pack(x, dc, dv):
    return np.concatenate([x.ravel(), dc.ravel(), dv.ravel()])

  def unpack(inputs):
    x_flat, dc_flat, dv_flat = np.split(inputs, [x.size, x.size + dc.size])
    return (x_flat.reshape(x.shape),
            dc_flat.reshape(dc.shape),
            dv_flat.reshape(dv.shape))

  def compute_xnew(inputs, lambda_):
    x, dc, dv = unpack(inputs)
    # avoid dividing by zero outside the design region
    dv = np.where(np.ravel(args['mask']) > 0, dv, 1)
    # square root is not defined for negative numbers, which can happen due to
    # small numerical errors in the computed gradients.
    xnew = x * np.maximum(-dc / (lambda_ * dv), 0) ** eta
    lower = np.maximum(0.0, x - max_move)
    upper = np.minimum(1.0, x + max_move)
    # note: autograd does not define gradients for np.clip
    return np.minimum(np.maximum(xnew, lower), upper)

  def f(inputs, lambda_):
    xnew = compute_xnew(inputs, lambda_)
    return volfrac - mean_density(xnew, args)

  # find_root allows us to differentiate through the while loop.
  inputs = pack(x, dc, dv)
  lambda_ = autograd_lib.find_root(f, inputs, lower_bound=1e-9, upper_bound=1e9)
  return compute_xnew(inputs, lambda_)


def sigmoid(x):
  return np.tanh(0.5*x)*.5 + 0.5


def logit(p):
  p = np.clip(p, 0, 1)
  return np.log(p) - np.log1p(-p)


# an alternative to the optimality criteria
def sigmoid_with_constrained_mean(x, average):
  f = lambda x, y: sigmoid(x + y).mean() - average
  lower_bound = logit(average) - np.max(x)
  upper_bound = logit(average) - np.min(x)
  b = autograd_lib.find_root(f, x, lower_bound, upper_bound)
  return sigmoid(x + b)


def calculate_forces(x_phys, args):
  applied_force = args['forces']

  if not args.get('g'):
    return applied_force

  density = 0
  for pad_left in [0, 1]:
    for pad_up in [0, 1]:
      padding = [(pad_left, 1 - pad_left), (pad_up, 1 - pad_up)]
      density += (1/4) * np.pad(
          x_phys.T, padding, mode='constant', constant_values=0
      )
  gravitional_force = -args['g'] * density[..., np.newaxis] * np.array([0, 1])
  return applied_force + gravitional_force.ravel()


def objective(x, ke, args, volume_contraint=False, cone_filter=True):
  """Objective function (compliance) for topology optimization."""
  kwargs = dict(penal=args['penal'], e_min=args['young_min'], e_0=args['young'])
  x_phys = physical_density(x, args, volume_contraint=volume_contraint,
                            cone_filter=cone_filter)
  forces = calculate_forces(x_phys, args)
  u = displace(
      x_phys, ke, forces, args['freedofs'], args['fixdofs'], **kwargs)
  c = compliance(x_phys, u, ke, **kwargs)
  return c


def optimality_criteria_step(x, ke, args):
  """Heuristic topology optimization, as described in the 88 lines paper."""
  c, dc = autograd.value_and_grad(objective)(x, ke, args)
  dv = autograd.grad(mean_density)(x, args)
  x = optimality_criteria_combine(x, dc, dv, args)
  return c, x


def run_toposim(x=None, args=None, loss_only=True, verbose=True):
  # Root function that runs the full optimization routine
  if args is None:
    args = default_args()
  if x is None:
    x = np.ones((args['nely'], args['nelx'])) * args['volfrac']  # init mass

  if not loss_only:
    frames = [x.copy()]
  ke = get_stiffness_matrix(args['young'], args['poisson'])  # stiffness matrix

  losses = []
  for step in range(args['opt_steps']):
    c, x = optimality_criteria_step(x, ke, args)
    losses.append(c)

    if not loss_only and verbose and step % 5 == 0:
      print('step {}, loss {:.2e}'.format(step, c))

    if not loss_only:
      frames.append(x.copy())

  return losses[-1] if loss_only else (losses, x, frames)

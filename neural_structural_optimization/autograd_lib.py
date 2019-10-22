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
# pylint: disable=unused-argument
# pylint: disable=g-import-not-at-top

import warnings

import autograd
import autograd.core
import autograd.extend
import autograd.numpy as anp
from neural_structural_optimization import caching
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
try:
  import sksparse.cholmod
  HAS_CHOLMOD = True
except ImportError:
  warnings.warn(
      'sksparse.cholmod not installed. Falling back to SciPy/SuperLU, but '
      'simulations will be about twice as slow.')
  HAS_CHOLMOD = False


# internal utilities
def _grad_undefined(_, *args):
  raise TypeError('gradient undefined for this input argument')


def _zero_grad(_, *args, **kwargs):
  def jvp(grad_ans):
    return 0.0 * grad_ans
  return jvp


# Gaussian filter
@autograd.extend.primitive
def gaussian_filter(x, width):
  """Apply gaussian blur of a given radius."""
  return scipy.ndimage.gaussian_filter(x, width, mode='reflect')


def _gaussian_filter_vjp(ans, x, width):
  del ans, x  # unused
  return lambda g: gaussian_filter(g, width)
autograd.extend.defvjp(gaussian_filter, _gaussian_filter_vjp)


# Cone filter
def _cone_filter_matrix(nelx, nely, radius, mask):
  x, y = np.meshgrid(np.arange(nelx), np.arange(nely), indexing='ij')

  rows = []
  cols = []
  values = []
  r_bound = int(np.ceil(radius))
  for dx in range(-r_bound, r_bound+1):
    for dy in range(-r_bound, r_bound+1):
      weight = np.maximum(0, radius - np.sqrt(dx**2 + dy**2))
      row = x + nelx * y
      column = x + dx + nelx * (y + dy)
      value = np.broadcast_to(weight, x.shape)

      # exclude cells beyond the boundary
      valid = (
          (mask > 0) &
          ((x+dx) >= 0) &
          ((x+dx) < nelx) &
          ((y+dy) >= 0) &
          ((y+dy) < nely)
      )
      rows.append(row[valid])
      cols.append(column[valid])
      values.append(value[valid])

  data = np.concatenate(values)
  i = np.concatenate(rows)
  j = np.concatenate(cols)
  return scipy.sparse.coo_matrix((data, (i, j)), (nelx * nely,) * 2)


@caching.ndarray_safe_lru_cache()
def normalized_cone_filter_matrix(nx, ny, radius, mask):
  """Calculate a sparse matrix appropriate for applying a cone filter."""
  raw_filters = _cone_filter_matrix(nx, ny, radius, mask).tocsr()
  weights = 1 / raw_filters.sum(axis=0).squeeze()
  diag_weights = scipy.sparse.spdiags(weights, 0, nx*ny, nx*ny)
  return (diag_weights @ raw_filters).tocsr()


@autograd.extend.primitive
def cone_filter(inputs, radius, mask=1, transpose=False):
  """Apply a cone filter of the given radius."""
  inputs = np.asarray(inputs)
  filters = normalized_cone_filter_matrix(
      *inputs.shape, radius=radius, mask=mask)
  if transpose:
    filters = filters.T
  outputs = filters @ inputs.ravel(order='F')
  return outputs.reshape(inputs.shape, order='F')


def _cone_filter_vjp(ans, inputs, radius, mask=1, transpose=False):
  del ans, inputs  # unused
  return lambda g: cone_filter(g, radius, mask, transpose=not transpose)
autograd.extend.defvjp(cone_filter, _cone_filter_vjp)


## a useful utility for 1D scatter operations
def inverse_permutation(indices):
  inverse_perm = np.zeros(len(indices), dtype=anp.int64)
  inverse_perm[indices] = np.arange(len(indices), dtype=anp.int64)
  return inverse_perm


# the 1D scatter operation
def scatter1d(nonzero_values, nonzero_indices, array_len):
  all_indices = np.arange(array_len, dtype=anp.int64)
  zero_indices = anp.setdiff1d(all_indices, nonzero_indices, assume_unique=True)
  index_map = inverse_permutation(
      anp.concatenate([nonzero_indices, zero_indices]))
  u_values = anp.concatenate([nonzero_values, anp.zeros(len(zero_indices))])
  return u_values[index_map]


@caching.ndarray_safe_lru_cache(1)
def _get_solver(a_entries, a_indices, size, sym_pos):
  """Get a solver for applying the desired matrix factorization."""
  # A cache size of one is sufficient to avoid re-computing the factorization in
  # the backwawrds pass.
  a = scipy.sparse.coo_matrix((a_entries, a_indices), shape=(size,)*2).tocsc()
  if sym_pos and HAS_CHOLMOD:
    return sksparse.cholmod.cholesky(a).solve_A
  else:
    # could also use scikits.umfpack.splu
    # should be about twice as slow as the cholesky
    return scipy.sparse.linalg.splu(a).solve


## Sparse solver
@autograd.primitive
def solve_coo(a_entries, a_indices, b, sym_pos=False):
  """Solve a sparse system of linear equations.

  Args:
    a_entries: numpy array with shape (num_zeros,) giving values for non-zero
      matrix entries.
    a_indices: numpy array with shape (2, num_zeros) giving x and y indices for
      non-zero matrix entries.
    b: 1d numpy array specifying the right hand side of the equation.
    sym_pos: is the matrix guaranteed to be positive-definite?

  Returns:
    1d numpy array corresponding to the solution of a*x=b.
  """
  solver = _get_solver(a_entries, a_indices, b.size, sym_pos)
  return solver(b)


# see autograd's np.linalg.solve:
# https://github.com/HIPS/autograd/blob/96a03f44da43cd7044c61ac945c483955deba957/autograd/numpy/linalg.py#L40


def solve_coo_adjoint(a_entries, a_indices, b, sym_pos=False):
  # NOTE: not tested on complex valued inputs.
  if sym_pos:
    return solve_coo(a_entries, a_indices, b, sym_pos)
  else:
    return solve_coo(a_entries, a_indices[::-1], b, sym_pos)


def grad_solve_coo_entries(ans, a_entries, a_indices, b, sym_pos=False):
  def jvp(grad_ans):
    lambda_ = solve_coo_adjoint(a_entries, a_indices, grad_ans, sym_pos)
    i, j = a_indices
    return -lambda_[i] * ans[j]
  return jvp


def grad_solve_coo_b(ans, a_entries, a_indices, b, sym_pos=False):
  def jvp(grad_ans):
    return solve_coo_adjoint(a_entries, a_indices, grad_ans, sym_pos)
  return jvp


autograd.extend.defvjp(
    solve_coo, grad_solve_coo_entries, _grad_undefined, grad_solve_coo_b)


@autograd.primitive
def find_root(
    f, x, lower_bound, upper_bound, tolerance=1e-12, max_iterations=64):
  # Implicitly solve f(x,y)=0 for y(x) using binary search.
  # Assumes that y is a scalar and f(x,y) is monotonic in y.
  for _ in range(max_iterations):
    y = 0.5 * (lower_bound + upper_bound)
    if upper_bound - lower_bound < tolerance:
      break
    if f(x, y) > 0:
      upper_bound = y
    else:
      lower_bound = y
  return y


def grad_find_root(y, f, x, lower_bound, upper_bound, tolerance=None):
  # This uses a special case of the adjoint gradient rule:
  # http://www.dolfin-adjoint.org/en/latest/documentation/maths/3-gradients.html#the-adjoint-approach
  def jvp(grad_y):
    g = lambda x: f(x, y)
    h = lambda y: f(x, y)
    return -autograd.grad(g)(x) / autograd.grad(h)(y) * grad_y
  return jvp


autograd.extend.defvjp(
    find_root, _grad_undefined, grad_find_root,
    _zero_grad, _zero_grad, _zero_grad)

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

# pylint: disable=g-complex-comprehension

import autograd.numpy
from autograd.test_util import check_grads
from neural_structural_optimization import autograd_lib
import numpy as np
from absl.testing import absltest

cone_filter = autograd_lib.cone_filter
gaussian_filter = autograd_lib.gaussian_filter
scatter1d = autograd_lib.scatter1d
solve_coo = autograd_lib.solve_coo
inverse_permutation = autograd_lib.inverse_permutation
find_root = autograd_lib.find_root


class AutogradLibTest(absltest.TestCase):

  def test_gaussian_filter(self):
    image = np.random.RandomState(0).rand(9, 9)
    width = 4
    np.testing.assert_allclose(gaussian_filter(image, width).sum(), image.sum())
    check_grads(lambda x: gaussian_filter(x, width), modes=['rev'])(image)

  def test_cone_filter(self):
    image = np.random.RandomState(0).rand(5, 5)
    width = 4
    check_grads(lambda x: cone_filter(x, width), modes=['rev'])(image)

  def test_inverse_permutation(self):
    indices = np.array([4, 2, 1, 7, 9, 5, 6, 0, 3, 8])
    inv_indices = inverse_permutation(indices)
    np.testing.assert_array_equal(np.array([7, 2, 1, 8, 0, 5, 6, 3, 9, 4]),
                                  inv_indices)

  def test_scatter1d(self):
    # also tests the `inverse_permutation` function
    nonzero_values = [4, 2, 7, 9]
    nonzero_indices = [2, 3, 7, 8]
    array_len = 10

    u = scatter1d(nonzero_values, nonzero_indices, array_len)
    np.testing.assert_array_equal(
        np.array([0., 0., 4., 2., 0., 0., 0., 7., 9., 0.]), u)

  def test_coo_solve(self):
    # test solve_coo gradients
    indices = np.array([[i % 10, (i - j) % 10]
                        for i in range(10) for j in range(-3, 4)]).T
    entries = np.random.RandomState(0).randn(indices.shape[-1])
    b = np.random.RandomState(0).rand(10)

    check_grads(lambda x: solve_coo(entries, indices, x), modes=['rev'])(b)
    check_grads(lambda x: solve_coo(x, indices, b), modes=['rev'])(entries)

  def test_find_root(self):
    # solve for a literal square root
    f = lambda x, y: y ** 2 - x
    result = find_root(f, 2, lower_bound=0, upper_bound=2)
    np.testing.assert_allclose(result, np.sqrt(2))

  def test_find_root_grad(self):
    f = lambda x, y: y ** 2 - abs(autograd.numpy.mean(x))
    x0 = np.random.RandomState(0).randn(3)
    check_grads(lambda x: find_root(f, x, 0, 10, 1e-12), modes=['rev'])(x0)

if __name__ == '__main__':
  absltest.main()

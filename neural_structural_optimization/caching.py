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
"""NumPy friendly caching utilities."""

import functools

import numpy as np


class _WrappedArray:
  """Hashable wrapper for NumPy arrays."""

  def __init__(self, value):
    self.value = value

  def __eq__(self, other):
    return np.array_equal(self.value, other.value)

  def __hash__(self):
    # Something that can be calculated quickly -- we won't have many collisions.
    # Hash collisions just mean that that __eq__ needs to be checked.
    # https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
    return hash(repr(self.value.ravel()))


def ndarray_safe_lru_cache(maxsize=128):
  """An ndarray compatible version of functools.lru_cache."""
  def decorator(func):  # pylint: disable=missing-docstring
    @functools.lru_cache(maxsize)
    def cached_func(*args, **kwargs):
      args = tuple(a.value if isinstance(a, _WrappedArray) else a for a in args)
      kwargs = {k: v.value if isinstance(v, _WrappedArray) else v
                for k, v in kwargs.items()}
      return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
      args = tuple(_WrappedArray(a) if isinstance(a, np.ndarray) else a
                   for a in args)
      kwargs = {k: _WrappedArray(v) if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()}
      return cached_func(*args, **kwargs)

    return wrapper
  return decorator

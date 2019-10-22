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
# pylint: disable=superfluous-parens
import functools

from absl import logging
import autograd
import autograd.numpy as np
from neural_structural_optimization import models
from neural_structural_optimization import topo_physics
import scipy.optimize
import tensorflow as tf
import xarray


def optimizer_result_dataset(losses, frames, save_intermediate_designs=False):
  # The best design will often but not always be the final one.
  best_design = np.nanargmin(losses)
  logging.info(f'Final loss: {losses[best_design]}')
  if save_intermediate_designs:
    ds = xarray.Dataset({
        'loss': (('step',), losses),
        'design': (('step', 'y', 'x'), frames),
    }, coords={'step': np.arange(len(losses))})
  else:
    ds = xarray.Dataset({
        'loss': (('step',), losses),
        'design': (('y', 'x'), frames[best_design]),
    }, coords={'step': np.arange(len(losses))})
  return ds


def train_tf_optimizer(
    model, max_iterations, optimizer, save_intermediate_designs=True,
):
  loss = 0
  model(None)  # build model, if not built
  tvars = model.trainable_variables

  losses = []
  frames = []
  for i in range(max_iterations + 1):
    with tf.GradientTape() as t:
      t.watch(tvars)
      logits = model(None)
      loss = model.loss(logits)

    losses.append(loss.numpy().item())
    frames.append(logits.numpy())

    if i % (max_iterations // 10) == 0:
      logging.info(f'step {i}, loss {losses[-1]:.2f}')

    if i < max_iterations:
      grads = t.gradient(loss, tvars)
      optimizer.apply_gradients(zip(grads, tvars))

  designs = [model.env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)


train_adam = functools.partial(
    train_tf_optimizer, optimizer=tf.keras.optimizers.Adam(1e-2))


def _set_variables(variables, x):
  shapes = [v.shape.as_list() for v in variables]
  values = tf.split(x, [np.prod(s) for s in shapes])
  for var, value in zip(variables, values):
    var.assign(tf.reshape(tf.cast(value, var.dtype), var.shape))


def _get_variables(variables):
  return np.concatenate([
      v.numpy().ravel() if not isinstance(v, np.ndarray) else v.ravel()
      for v in variables])


def train_lbfgs(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
    **kwargs
):
  model(None)  # build model, if not built

  losses = []
  frames = []

  if init_model is not None:
    if not isinstance(model, models.PixelModel):
      raise TypeError('can only use init_model for initializing a PixelModel')
    model.z.assign(tf.cast(init_model(None), model.z.dtype))

  tvars = model.trainable_variables

  def value_and_grad(x):
    _set_variables(tvars, x)
    with tf.GradientTape() as t:
      t.watch(tvars)
      logits = model(None)
      loss = model.loss(logits)
    grads = t.gradient(loss, tvars)
    frames.append(logits.numpy().copy())
    losses.append(loss.numpy().copy())
    return float(loss.numpy()), _get_variables(grads).astype(np.float64)

  x0 = _get_variables(tvars).astype(np.float64)
  # rely upon the step limit instead of error tolerance for finishing.
  _, _, info = scipy.optimize.fmin_l_bfgs_b(
      value_and_grad, x0, maxfun=max_iterations, factr=1, pgtol=1e-14, **kwargs
  )
  logging.info(info)

  designs = [model.env.render(x, volume_contraint=True) for x in frames]
  return optimizer_result_dataset(
      np.array(losses), np.array(designs), save_intermediate_designs)


def constrained_logits(init_model):
  """Produce matching initial conditions with volume constraints applied."""
  logits = init_model(None).numpy().astype(np.float64).squeeze(axis=0)
  return topo_physics.physical_density(
      logits, init_model.env.args, volume_contraint=True, cone_filter=False)


def method_of_moving_asymptotes(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
):
  import nlopt  # pylint: disable=g-import-not-at-top

  if not isinstance(model, models.PixelModel):
    raise ValueError('MMA only defined for pixel models')

  env = model.env
  if init_model is None:
    x0 = _get_variables(model.trainable_variables).astype(np.float64)
  else:
    x0 = constrained_logits(init_model).ravel()

  def objective(x):
    return env.objective(x, volume_contraint=False)

  def constraint(x):
    return env.constraint(x)

  def wrap_autograd_func(func, losses=None, frames=None):
    def wrapper(x, grad):
      if grad.size > 0:
        value, grad[:] = autograd.value_and_grad(func)(x)
      else:
        value = func(x)
      if losses is not None:
        losses.append(value)
      if frames is not None:
        frames.append(env.reshape(x).copy())
      return value
    return wrapper

  losses = []
  frames = []

  opt = nlopt.opt(nlopt.LD_MMA, x0.size)
  opt.set_lower_bounds(0.0)
  opt.set_upper_bounds(1.0)
  opt.set_min_objective(wrap_autograd_func(objective, losses, frames))
  opt.add_inequality_constraint(wrap_autograd_func(constraint), 1e-8)
  opt.set_maxeval(max_iterations + 1)
  opt.optimize(x0)

  designs = [env.render(x, volume_contraint=False) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)


def optimality_criteria(
    model, max_iterations, save_intermediate_designs=True, init_model=None,
):
  if not isinstance(model, models.PixelModel):
    raise ValueError('optimality criteria only defined for pixel models')

  env = model.env
  if init_model is None:
    x = _get_variables(model.trainable_variables).astype(np.float64)
  else:
    x = constrained_logits(init_model).ravel()

  # start with the first frame but not its loss, since optimality_criteria_step
  # returns the current loss and the *next* design.
  losses = []
  frames = [x]
  for _ in range(max_iterations):
    c, x = topo_physics.optimality_criteria_step(x, env.ke, env.args)
    losses.append(c)
    if np.isnan(c):
      # no point in continuing to optimize
      break
    frames.append(x)
  losses.append(env.objective(x, volume_contraint=False))

  designs = [env.render(x, volume_contraint=False) for x in frames]
  return optimizer_result_dataset(np.array(losses), np.array(designs),
                                  save_intermediate_designs)


def train_batch(model_list, flag_values, train_func=train_adam):
  batch_hist = []
  for batch_ix in range(flag_values.trials):
    logging.info(f'Starting trial {batch_ix}')
    history = train_func(model_list[batch_ix], flag_values)
    batch_hist.append(history)

  batch_hist = xarray.concat(batch_hist, dim='batch')
  return batch_hist

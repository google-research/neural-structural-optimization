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
import os.path

from absl import flags
from absl.testing import flagsaver
import apache_beam as beam
from neural_structural_optimization import pipeline
import tensorflow as tf

from absl.testing import absltest

FLAGS = flags.FLAGS


class PipelineTest(absltest.TestCase):

  def test(self):
    with flagsaver.flagsaver(
        num_seeds=2,
        optimization_steps=10,
        save_dir=FLAGS.test_tmpdir,
        experiment_name='unittest',
        problem_filter='l_shape',  # should use the mask
        quick_run=True,
        cnn_kwargs='offset_scale=1.0',
    ):
      pipeline.main([], runner=beam.runners.DirectRunner())

    # verify that at least one file was written
    path = os.path.join(
        FLAGS.test_tmpdir,
        'unittest/l_shape_0.2_64x64_0.4/pixels-oc_best.png',
    )
    self.assertTrue(tf.io.gfile.exists(path))

  def test_dynamic(self):
    with flagsaver.flagsaver(
        num_seeds=2,
        optimization_steps=1,
        save_dir=FLAGS.test_tmpdir,
        experiment_name='unittest',
        problem_filter='cantilever_beam',
        quick_run=True,
        dynamic_depth=True,
        cnn_kwargs='latent_scale=1;offset_scale=0',
    ):
      pipeline.main([], runner=beam.runners.DirectRunner())


if __name__ == '__main__':
  absltest.main()

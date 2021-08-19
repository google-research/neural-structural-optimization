from neural_structural_optimization.models import *

def FlatConv3(inputs, filters, activation=layers.Activation(tf.nn.relu)):
  he_normal = tf.keras.initializers.HeNormal()
  net = layers.Conv2D(filters, 3, padding='same', kernel_initializer=he_normal)(inputs)
  net = global_normalization(net)
  return activation(net)

def UpConv4(inputs, filters, activation=layers.Activation(tf.nn.relu)):
    he_normal = tf.keras.initializers.HeNormal()
#     net = layers.UpSampling2D(size=(2,2), interpolation='nearest')(inputs)
#     net = layers.Conv2D(filters, 4, strides=(1, 1), padding='same', kernel_initializer=he_normal)(net)
    net = layers.Conv2DTranspose(filters, 4, strides=(2,2),padding='same', kernel_initializer=he_normal)(inputs)
    net = global_normalization(net)
    return activation(net)

def DownConv4(inputs, filters, activation=layers.Activation(tf.nn.relu)):
    he_normal = tf.keras.initializers.HeNormal()
    net = layers.Conv2D(filters, 4, strides=(2, 2), padding='same', kernel_initializer=he_normal)(inputs)
    net = global_normalization(net)
    return activation(net)

class CNNSketch(Model):

  def __init__(
      self,
      seed=0,
      args=None,
      latent_scale=1.0,
      activation = tf.nn.relu
  ):
    super().__init__(seed, args)

    activation = layers.Activation(activation)

    h = self.env.args['nely'] // 8
    w = self.env.args['nelx'] // 8
    
    net = inputs = layers.Input((h, w, 256), batch_size=1)  ## 256 * h/8 * w/8

    net = activation(net)
    
    net = FlatConv3(net, 256, activation)

    net = UpConv4(net, 256, activation)  # h/4 * w/4
    net = FlatConv3(net, 256, activation)
    net = FlatConv3(net, 128, activation)

    net = UpConv4(net, 128, activation)  # h/2 * w/2
    net = FlatConv3(net, 128, activation)
    net = FlatConv3(net, 48, activation)

    net = UpConv4(net, 48, activation)  # h * w
    net = FlatConv3(net, 24, activation)
    net = layers.Conv2D(1, 3, padding='same')(net)
    # net = global_normalization(net)
    # net = tf.nn.sigmoid(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')

  def call(self, inputs=None):
    return self.core_model(self.z)
  
class Pix2PixCNNSketch(Model):

  def __init__(
      self,
      seed=0,
      args=None,
      latent_scale=1.0,
      activation = tf.nn.relu
  ):
    super().__init__(seed, args)

    activation = layers.Activation(activation)

    h = self.env.args['nely'] 
    w = self.env.args['nelx'] 
    
    net = inputs = layers.Input((h, w, 1), batch_size=1)  ## h * w

    net = DownConv4(net, 32, activation)  # h/2 * w/2
    net = FlatConv3(net, 64, activation)  
    net = FlatConv3(net, 64, activation)
    
    net = DownConv4(net, 64, activation)  # h/4 * w/4
    net = FlatConv3(net, 128, activation)
    net = FlatConv3(net, 128, activation)
    
    net = UpConv4(net, 128, activation)  # h/2 * w/2
    net = FlatConv3(net, 128, activation)  
    net = FlatConv3(net, 64, activation)

    net = UpConv4(net, 48, activation)  # h * w
    net = FlatConv3(net, 24, activation)
    net = layers.Conv2D(1, 3, padding='same')(net)

    outputs = tf.squeeze(net, axis=[-1])

    self.core_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    latent_initializer = tf.initializers.RandomNormal(stddev=latent_scale)
    self.z = self.add_weight(
        shape=inputs.shape, initializer=latent_initializer, name='z')

  def call(self, inputs=None):
    return self.core_model(self.z)

def main():
    from neural_structural_optimization import problems
    from neural_structural_optimization import topo_api
    problem = problems.mbb_beam(width=64, height=32, density=0.5)
    args = topo_api.specified_task(problem)
    cnns = CNNSketch(args=args)
    print("Create CNNSketch")

if __name__== "__main__" :
    main()
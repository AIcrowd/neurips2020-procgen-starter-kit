from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf

from ray.rllib.models import ModelCatalog

tf = try_import_tf()

"""
NOTE : This implementation has been taken from : 
    https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py

    to act as a reference implementation for implementing custom models.
"""


def get_conv_activation(model_config):
    if model_config.get("conv_activation") == "linear":
        activation = None
    else:
        activation = getattr(tf.nn, model_config.get("conv_activation"))
    return activation


def get_fc_activation(model_config):
    activation = model_config.get("fcnet_activation")
    if activation is None:
        activation = tf.keras.layers.ReLU()
    return activation


def conv_layers(x, model_config, obs_space, prefix=""):
    filters = model_config.get("conv_filters")
    if not filters:
        filters = _get_filter_config(obs_space.shape)

    activation = get_conv_activation(model_config)

    for i, (out_size, kernel, stride) in enumerate(filters, 1):
        x = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="same",
            data_format="channels_last",
            name=f"{prefix}conv{i}",
        )(x)
    return x


def fc_layers(x, model_config, prefix=""):
    x = tf.keras.layers.Flatten()(x)
    activation = get_fc_activation(model_config)
    fc_layers_config = model_config.get("fcnet_hiddens", [])
    for i, dim in enumerate(fc_layers_config):
        x = tf.keras.layers.Dense(
            units=dim, activation=activation, name=f"{prefix}fc-{i}"
        )(x)
    return x


def get_final_fc(x, num_outputs, model_config):
    x = tf.keras.layers.Dense(num_outputs, name="pi")(x)
    return x


def value_layers(x, inputs, obs_space, model_config):
    if not model_config.get("vf_share_layers"):
        x = conv_layers(inputs, model_config, obs_space, prefix="vf-")
        x = fc_layers(x, model_config, prefix="vf-")
    x = tf.keras.layers.Dense(units=1, name="vf")(x)
    return x


class MyVisionNetwork(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyVisionNetwork, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        last_layer = inputs
        # Build the conv layers
        last_layer = conv_layers(last_layer, model_config, obs_space)
        # Build the linear layers
        last_layer = fc_layers(last_layer, model_config)
        # Final linear layer
        logits = get_final_fc(last_layer, num_outputs, model_config)
        # Build the value layers
        value_out = value_layers(last_layer, inputs, obs_space, model_config)

        self.base_model = tf.keras.Model(inputs, [logits, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        logits, self._value_out = self.base_model(
            tf.cast(input_dict["obs"], tf.float32)
        )
        return logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


# Register model in ModelCatalog
ModelCatalog.register_custom_model("my_vision_network", MyVisionNetwork)

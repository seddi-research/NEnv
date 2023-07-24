import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
from torchvision import transforms

scaler = torch.cuda.amp.GradScaler()
import warnings

warnings.filterwarnings("ignore")
from nsf import nn as nn_
from nsf import utils
from nsf.nde import flows, transforms
from nsf.nde.distributions.uniform import TweakedUniform

from nsf.nde.distributions.normal import StandardNormal

from NEnv.Utils.utils import get_pdf_environment_map
from NEnv.Utils.EnvironmentMap import Envmap

TMP_DIR = "nenv/"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NEnv():
    _suffix = "nenv"
    _config_extension = ".nenv"
    _artifact_name = "nenv"

    @property
    def class_name(self):
        return self.__class__.__name__

    @staticmethod
    def _compute_unique_timestamp():
        time.sleep(1)
        ts = time.time()
        return datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

    def __init__(self,
                 path_hdr,
                 target_resolution=(4000, 2000),
                 batch_size=50000,
                 lr=5e-4,
                 depth=1,
                 number_coupling=2,
                 width=128,
                 bins=128,
                 model_name='Spline',
                 epochs=10000,
                 prior='Uniform',
                 batch_norm=True,
                 iterations_val=20,
                 wandb_run=None,
                 load_envmap=True,
                 ):
        """
        @param path_hdr: Path to the input HDR you want to learn from
        @type path_hdr: str
        @param target_resolution: Target Resolution used to sample the envmap
        @type target_resolution: tuple
        @param batch_size: Batch size for training
        @type batch_size: int
        @param lr: Learning rate for trianing
        @type lr: float
        @param depth: Number of hidden layers in each coupling layer
        @type depth: int
        @param number_coupling: Number of coupling layers in the flow
        @type number_coupling: int
        @param width: Number of hidden neurons in each coupling layer
        @type width: int
        @param bins: Number of bins in each coupling layer. O
        @type bins: int
        @param model_name: Model name. In: 'RealNVP', 'Muller_Linear', 'Muller_Quadratic', 'Spline', 'Cubic'
        @type model_name: str
        @param epochs: Number of epochs to train the model
        @type epochs: int
        @param prior: Prior distribution for the model. We currently only support 'Uniform'
        @type prior: str
        @param batch_norm: Set to True if you want to use batch normalization
        @type batch_norm: bool
        @param iterations_val: Number of iterations to compute validation results
        @type iterations_val: int
        @param wandb_run: Wandb run
        @type wandb_run: object
        @param load_envmap: Set to True if you want to pre-process the environment map (only recommended for training)
        @type load_envmap: bool
        """

        assert model_name in ['RealNVP', 'Muller_Linear', 'Muller_Quadratic', 'Spline', 'Cubic']

        self._path_hdr = path_hdr
        self._lr = lr
        self._target_resolution = target_resolution
        self._batch_size = int(batch_size)
        self._depth = depth
        self._number_coupling = number_coupling
        self._width = width
        self._bins = bins
        self._epochs = epochs
        self._prior = prior
        self._batch_norm = batch_norm
        self._iterations_val = iterations_val
        self.wandb_run = wandb_run

        self._model_name = model_name
        self._temporary_model_name = os.path.join(
            TMP_DIR, model_name + ".pth"
        )

        self._device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if load_envmap:
            print('Reading Environment Map')
            self.envmap = Envmap(path_hdr, gamma=1, resolution=target_resolution)

            print('Computing PDF of GT environment Map')
            self._gt_pdf = get_pdf_environment_map(self.envmap)

    def create_base_transform(self, i, model_name, hidden_features, depth, batch_norm, bins):
        """
        Method that creates a normalizing flow coupling layer based on the desired configuration

        @param i: Layer number
        @type i: int
        @param model_name: Type of flow
        @type model_name: str
        @param hidden_features: Number of hidden neurons in the flow MLP
        @type hidden_features: int
        @param depth: Number of hidden layers in the flow MLP
        @type depth: int
        @param batch_norm: Set to true if you want batch norm in this layer
        @type batch_norm: bool
        @param bins: Number of bins in the layer. Not applicable for RealNVP
        @type bins: int
        @return: layer
        @rtype: object
        """

        if model_name == 'RealNVP':
            return transforms.AffineCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=2,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=depth,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                ),
            )
        elif model_name == 'Muller_Linear':
            return transforms.PiecewiseLinearCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=2,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=depth,
                    dropout_probability=0.0,
                    use_batch_norm=batch_norm,
                ),
                num_bins=bins,
            )

        elif model_name == 'Muller_Quadratic':
            return transforms.PiecewiseQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=2,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=depth,
                    dropout_probability=0.0,
                    use_batch_norm=batch_norm,
                ),
                num_bins=bins,
            )

        elif model_name == 'Cubic':
            return transforms.PiecewiseCubicCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=2,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=depth,
                    dropout_probability=0.0,
                    use_batch_norm=batch_norm,
                ),
                num_bins=bins,
            )

        elif model_name == 'Spline':
            return transforms.PiecewiseQuadraticCouplingTransform(
                mask=utils.create_alternating_binary_mask(
                    features=2,
                    even=(i % 2 == 0)
                ),
                transform_net_create_fn=lambda in_features, out_features:
                nn_.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    num_blocks=depth,
                    dropout_probability=0.0,
                    use_batch_norm=batch_norm,
                ),
                num_bins=bins,
            )

    def define_model(self, pre_trained_network=None):
        """

        @param pre_trained_network: Path to pretrained model weights, if available
        @type pre_trained_network: str
        @return: Initialized model
        @rtype: object
        """

        if self._model_name != 'RealNVP':
            distribution = TweakedUniform(
                low=torch.zeros(2),
                high=torch.ones(2)
            )
        else:
            distribution = StandardNormal([2])
        distribution._validate_args = False
        transform = transforms.CompositeTransform([
            self.create_base_transform(i, model_name=self._model_name, batch_norm=self._batch_norm, bins=self._bins,
                                       hidden_features=self._width, depth=self._depth) for i in
            range(self._number_coupling)
        ])
        flow = flows.Flow(transform, distribution).to(self._device)
        if pre_trained_network is not None:
            model_weights = torch.load(pre_trained_network)
            flow.load_state_dict(model_weights, strict=True)

        return flow

    def set_model_path(self, model_path: str):
        """
        Method that updates the path to the generator in disc.

        :param model_path: New generator path.
        """
        self._model_path = model_path

    def clean_up(self):
        """
        Method that cleans CUDA memory after execution.
        """
        try:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            shutil.rmtree(self._temporary_model_name, ignore_errors=True)
        except:
            print("No torch memory to clean")

    @property
    def class_name(self):
        return self.__class__.__name__

    def export(self, output_path):
        """

        :param output_path: Path to store this feature into
        :return: feature path and config filename
        """
        print("Exporting {} feature: ".format(self.class_name))

        self.timestamp = self._compute_unique_timestamp()
        self.config_filename = (self.timestamp + self._config_extension
                                )
        self.folder_name = self.timestamp + self._suffix
        self.feature_path = os.path.join(output_path, self.folder_name)

        # Create the output folder for this feature if not exists
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)

        flow_network_filename = "flow.pth"
        flow_network_path = os.path.join(
            self.feature_path, flow_network_filename
        )
        try:
            torch.save(self._model.state_dict(), flow_network_path)
        except Exception as e:
            print("No generator has been trained yet! ")

        config = {
            "timestamp": self.timestamp,
            "lr": self._lr,
            "epochs": self._epochs,
            "depth": self._depth,
            "width": self._width,
            "model_name": self._model_name,
            "bins": self._bins,
            "prior": self._prior,
            "model_path": flow_network_path,
            "batch_norm": self._batch_norm,
            "iterations_val": self._iterations_val,
            "path_hdr": self._path_hdr,
            "target_resolution": self._target_resolution,
            "number_coupling": self._number_coupling,
        }
        with open(
                os.path.join(self.feature_path, self.config_filename), "w"
        ) as outfile:
            json.dump(
                config,
                outfile,
                separators=(",", ":"),
                sort_keys=True,
                indent=4,
                cls=NumpyEncoder,
            )

        if self.wandb_run:
            import wandb
            trained_model_artifact = wandb.Artifact("Nenv", type="flow")
            trained_model_artifact.add_dir(self.feature_path)
            self.wandb_run.log_artifact(trained_model_artifact)

        return self.feature_path, self.config_filename

    def load(self, config_path):
        """
        Load an existing execution of a transfer network. This method loads a file (.transfer*) in JSON format,
        containing the path with previous results obtained with the parameters writen inside.
        :param config_path: Configuration filename to be read
        :return: Nothing
        """
        self.feature_path = os.path.dirname(config_path)

        data = json.loads(open(config_path).read())
        self.timestamp = data["timestamp"]
        self._path_hdr = data['path_hdr']
        self._target_resolution = data['target_resolution']
        self._number_coupling = data['number_coupling']

        self._lr = data["lr"]
        self._epochs = data["epochs"]
        self._depth = data["depth"]
        self._width = data["width"]
        self._model_name = data["model_name"]
        self._bins = data["bins"]
        self._prior = data["prior"]
        self._batch_norm = data["batch_norm"]
        self._iterations_val = data["iterations_val"]
        self._model_path = data["model_path"]

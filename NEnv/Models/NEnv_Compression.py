import datetime
import json
import os
import shutil
import time

import numpy as np
import torch

scaler = torch.cuda.amp.GradScaler()
import warnings

warnings.filterwarnings("ignore")
from NEnv.Architectures.SIREN import Siren

from NEnv.Utils.utils import get_gt_image
from NEnv.Utils.EnvironmentMap import Envmap

TMP_DIR = "nenv/"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NEnv_Compression():
    _suffix = "nenv_compression"
    _config_extension = ".nenv_compression"
    _artifact_name = "nenv_compression"

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
                 target_resolution=(2000, 1000),
                 batch_size=300000,
                 lr=5e-4,
                 depth=1,
                 width=128,
                 model_name='SIREN',
                 epochs=10000,
                 loss_function='L1',
                 batch_norm=True,
                 iterations_val=10,
                 proportional=False,
                 wandb_run=None,
                 load_envmap=True,
                 step_size_scheduler=1000,
                 gamma=2.5,
                 ):

        assert model_name in ['SIREN', ]  # TODO introduce more models
        assert loss_function in ['L1', 'L2']
        assert batch_size > 0
        assert lr > 0
        assert depth > 0
        assert width > 0

        self._path_hdr = path_hdr
        self._lr = lr
        self._target_resolution = target_resolution
        self._batch_size = int(batch_size)
        self._depth = int(depth)
        self._width = int(width)
        self._epochs = epochs
        self._loss_function = loss_function
        self._batch_norm = batch_norm
        self._iterations_val = iterations_val
        self._step_size_scheduler = step_size_scheduler
        self._proportional = proportional
        self._gamma = gamma
        self.wandb_run = wandb_run

        self._model_name = model_name
        """if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)"""
        self._temporary_model_name = os.path.join(
            TMP_DIR, model_name + ".pth"
        )

        self._device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        if load_envmap:
            print('Reading Environment Map')
            self.envmap = Envmap(path_hdr, gamma=1, resolution=target_resolution)
            self._gt_im = get_gt_image(self.envmap)

    def define_model(self, pre_trained_network=None):

        if self._model_name == 'SIREN':
            model = Siren(in_features=2, out_features=3, hidden_features=self._width, hidden_layers=self._depth,
                          outermost_linear=True)
        else:
            raise Exception('Model type not implemented')
        if pre_trained_network is not None:
            model_weights = torch.load(pre_trained_network)
            model.load_state_dict(model_weights, strict=True)

        return model

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

        compressor_network_filename = "compressor.pth"
        compressor_network_path = os.path.join(
            self.feature_path, compressor_network_filename
        )
        try:
            torch.save(self._model.state_dict(), compressor_network_path)
        except Exception as e:
            print("No generator has been trained yet! ")

        config = {
            "timestamp": self.timestamp,
            "lr": self._lr,
            "batch_size": self._batch_size,
            "proportional": self._proportional,
            "gamma": self._gamma,
            "step_size_scheduler": self._step_size_scheduler,
            "loss": self._loss_function,
            "epochs": self._epochs,
            "depth": self._depth,
            "width": self._width,
            "model_name": self._model_name,
            "model_path": compressor_network_path,
            "batch_norm": self._batch_norm,
            "iterations_val": self._iterations_val,
            "path_hdr": self._path_hdr,
            "target_resolution": self._target_resolution,
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
            trained_model_artifact = wandb.Artifact("Nenv_compressor", type="compressor")
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
        self._lr = data["lr"]
        self._epochs = data["epochs"]
        self._depth = data["depth"]
        self._width = data["width"]
        self._model_name = data["model_name"]
        self._batch_norm = data["batch_norm"]
        self._iterations_val = data["iterations_val"]
        self._model_path = data["model_path"]
        self._proportional = data["proportional"]
        self._gamma = data["gamma"]
        self._loss_function = data["loss"]
        self._batch_size = data["batch_size"]

import argparse

import wandb

from NEnv.Models.NEnv import NEnv
import warnings

warnings.filterwarnings("ignore")


def _parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    arguments = _parse_args()


    run = wandb.init(project='NEnv', job_type="training", notes="NEnv",config={})


    # Change this to the path of the envmap you want to train a nenv with
    path_env = r"whale_skeleton_4k.hdr"


    nenv = NEnv(path_hdr = path_env,
                wandb_run=run,
                target_resolution=(4000, 2000),
                depth=2,
                number_coupling=2,
                width=256,
                bins=256,
                model_name='Spline',
                epochs=100000,
                )

    nenv.compute()

    # Change this to the path where you want to save your nenv
    nenv.export("path_export")


if __name__ == '__main__':
    main()
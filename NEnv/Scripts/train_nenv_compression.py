import argparse
import warnings

import wandb

from NEnv.Models.NEnv_Compression import NEnv_Compression

warnings.filterwarnings("ignore")


def _parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    arguments = _parse_args()

    run = wandb.init(project='NEnv', job_type="training", notes="NEnv", config={})

    # Change this to the path of the envmap you want to train a nenv with
    path_env = r"whale_skeleton_4k.hdr"

    nenv = NEnv_Compression(path_hdr=path_env,
                            wandb_run=run,
                            target_resolution=(4000, 2000),
                            depth=3,
                            width=512,
                            model_name='SIREN',
                            epochs=10000,
                            batch_size=300000,
                            step_size_scheduler=2000,
                            proportional=False,
                            iterations_val=100,
                            )

    nenv.compute()

    # Change this to the path where you want to save your nenv
    nenv.export("path_export")


if __name__ == '__main__':
    main()

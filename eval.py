import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
import torch
import yaml
import numpy as np

from dataset import Dataloader
from model import GNNModel
from run import compute_model_output
from tools import (
    verbose_model,
)


def eval(
    data,
    model,
    criterion,
    optimizer,
    device,
    verbose,
    dire_path,
    pred_only,
):
    verbose = verbose if verbose == 0 or verbose == 1 else 2
    nb_batch = len(data.main_loader)

    model.eval()
    avg_loss, avg_metrics = compute_model_output(
        data.main_loader,
        model,
        criterion,
        optimizer,
        device,
        nb_batch,
        epoch=1,
        num_epochs=1,
        eval=True,
        output_pred=True,
        output_dire=dire_path,
        pred_only=pred_only,
    )

    if not pred_only:
        print("")
        verbose_model(1, avg_loss, avg_metrics, "Evaluation", verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GNN Model with specified configuration"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration YAML file during training",
        default="config.yaml",
    )
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the dataset folder",
        default="/validation",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path of the predicted loads.npy",
        default="/logs/models",
    )
    parser.add_argument(
        "verbose",
        type=int,
        default=2,
    )
    parser.add_argument("pred_only", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    with open(args.config_path, "r") as file:
        hparams = yaml.safe_load(file)

    np.random.seed(hparams["hyperparameters"]["seed"])
    torch.manual_seed(hparams["hyperparameters"]["seed"])

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hparams["hyperparameters"]["seed"])

    data = Dataloader(
        args.dir_path,
        lag_step=hparams["data"]["lag_step"],
        lag_jump=hparams["data"]["lag_jump"],
        log_edge_attr=hparams["data"]["log_edge_attr"],
        batch_size=1,
        train_percent=1,
        shuffle_main=False,
        adjacency_file="/adjacency.json",
        injections_file="/injections.npy",
        loads_file="/loads.npy",
    )

    model = GNNModel(
        input_features=hparams["model"]["input_features"],
        hidden_features=hparams["model"]["hidden_features"],
        output_features=hparams["model"]["output_features"],
        num_layers=hparams["model"]["num_layers"],
        dropout=hparams["model"]["dropout"],
        leaky_or_tanh=hparams["model"]["leaky_or_tanh"],
        use_gat=hparams["model"]["use_gat"],
        use_attention=hparams["model"]["use_attention"],
    )

    criterion = torch.nn.MSELoss()
    optimizer = None

    logdir = hparams["folder_file"]["path_to_model_dir"]

    device = torch.device("cpu")
    if hparams["folder_file"]["device"] != "cpu" and torch.cuda.is_available():
        device = torch.device(hparams["folder_file"]["device"])
    print(f"device is set to {device}")

    eval(
        data,
        model,
        criterion,
        optimizer,
        device,
        args.verbose,
        args.output_path,
        args.pred_only,
    )

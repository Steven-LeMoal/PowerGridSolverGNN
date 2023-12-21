import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import argparse
import datetime
import torch
import yaml
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from dataset import Dataloader
from model import GNNModel
from run import compute_model_output
from tools import (
    EarlyStopping,
    flatten_config,
    log_metrics,
    save_plot_loss_and_metrics,
    verbose_model,
)


def update_config_and_save(hparams, new_folder_path, new_logdir):
    hparams["folder_file"]["path_to_model_dir"] = new_folder_path

    if not os.path.exists(new_logdir):
        os.makedirs(new_logdir)

    new_config_path = os.path.join(new_logdir, "config.yaml")

    with open(new_config_path, "w") as file:
        yaml.dump(hparams, file)

    return new_config_path


def train(
    data,
    model,
    hparams,
    criterion,
    optimizer,
    early_stopping,
    writer,
    device,
    verbose,
    save_plot,
    logdir,
):
    verbose = verbose if verbose == 0 or verbose == 1 else 2
    best_val_loss = float("inf")
    num_epochs = hparams["hyperparameters"]["num_epochs"]
    monitor_val_loss = (
        hparams["hyperparameters"]["monitor_val_loss_or_train_loss"] != "train_loss"
    )
    train_nb_batch = len(data.main_loader)
    val_nb_batch = len(data.val_loader)

    train_loss, train_metrics = [], []
    val_loss, val_metrics = [], []

    for epoch in range(1, num_epochs + 1):
        avg_train_loss, avg_train_metrics = compute_model_output(
            data.main_loader,
            model,
            criterion,
            optimizer,
            device,
            train_nb_batch,
            epoch,
            num_epochs,
            eval=False,
            output_pred=False,
            output_dire=None,
        )

        train_loss.append(avg_train_loss)
        train_metrics.append(avg_train_metrics)

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        log_metrics(writer, "train", avg_train_metrics, epoch)

        verbose_model(epoch, avg_train_loss, avg_train_metrics, "Training", verbose)

        model.eval()
        avg_val_loss, avg_val_metrics = compute_model_output(
            data.val_loader,
            model,
            criterion,
            optimizer,
            device,
            val_nb_batch,
            epoch,
            num_epochs,
            eval=False,
            output_pred=False,
            output_dire=None,
        )

        val_loss.append(avg_val_loss)
        val_metrics.append(avg_val_metrics)

        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        log_metrics(writer, "val", avg_val_metrics, epoch)

        verbose_model(epoch, avg_val_loss, avg_val_metrics, "Validation", verbose)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{logdir}/best_model_checkpoint.pth")
            print(f"Improved best validation loss at epoch {epoch} : saving...")

        current_score = avg_val_loss if monitor_val_loss else avg_train_loss
        early_stopping(current_score)

        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        print("\n ----------------- \n")

    writer.add_hparams(
        hparam_dict=flatten_config(hparams),
        metric_dict={"best_val_loss": best_val_loss},
    )

    update_config_and_save(hparams, f"{logdir}/best_model_checkpoint.pth", logdir)

    writer.close()

    if save_plot:
        save_plot_loss_and_metrics(
            train_loss, train_metrics, train=True, save_dir=logdir
        )
        save_plot_loss_and_metrics(val_loss, val_metrics, train=False, save_dir=logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN Model with specified configuration"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the configuration YAML file",
        default="/config.yaml",
    )
    parser.add_argument(
        "dir_path",
        type=str,
        help="Path to the dataset folder",
        default="/datasets",
    )
    parser.add_argument(
        "verbose",
        type=int,
        default=2,
    )
    parser.add_argument(
        "save_plot",
        action=argparse.BooleanOptionalAction,
    )
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
        batch_size=hparams["hyperparameters"]["batchsize"],
        train_percent=hparams["hyperparameters"]["train_percent"],
        shuffle_main=hparams["hyperparameters"]["shuffle_train"],
        shuffle_val=hparams["hyperparameters"]["shuffle_val"],
        log_edge_attr=hparams["data"]["log_edge_attr"],
        adjacency_file="/adjacency.json",
        injections_file="/injections.npy",
        loads_file="/loads.npy",
    )

    hparams["model"]["input_features"] = hparams["data"]["lag_step"] + 1

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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["hyperparameters"]["learning_rate"]
    )
    early_stopping = EarlyStopping(
        patience=hparams["hyperparameters"]["patience"],
        monitor=hparams["hyperparameters"]["monitor_val_loss_or_train_loss"],
    )

    cur_dir = (
        os.curdir
        if hparams["folder_file"]["log_folder_in_dire"] != ""
        else hparams["folder_file"]["log_folder_in_dire"]
    )

    logdir = os.path.join(
        cur_dir + hparams["folder_file"]["log_folder_in_dire"],
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    print(f"\nThe trained model and other data will be saved here : {logdir}\n")
    writer = SummaryWriter(logdir)

    device = torch.device("cpu")
    if hparams["folder_file"]["device"] != "cpu" and torch.cuda.is_available():
        device = torch.device(hparams["folder_file"]["device"])
    print(f"device is set to {device}")

    train(
        data,
        model,
        hparams,
        criterion,
        optimizer,
        early_stopping,
        writer,
        device,
        args.verbose,
        args.save_plot,
        logdir,
    )

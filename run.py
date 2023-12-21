import datetime
import os
import torch
import numpy as np
from tools import compute_metrics
from tqdm import tqdm


def compute_model_output(
    data,
    model,
    criterion,
    optimizer,
    device,
    nb_batch,
    epoch,
    num_epochs,
    eval=False,
    output_pred=False,
    output_dire=None,
    pred_only=False,
):
    total_loss = 0
    total_metrics = {
        "MAE": 0,
        "RMSE": 0,
        "MAPE": 0,
        "R_SQUARED": 0,
        "MAX_ERROR": 0,
    }

    all_predictions = []

    if eval:
        with torch.no_grad():
            if pred_only:
                for batch in tqdm(
                    data, desc=f"Evaluation (epoch {epoch}/{num_epochs}): "
                ):
                    output = model(batch.to(device))
                    all_predictions.append(output)
            else:
                for batch in tqdm(
                    data, desc=f"Evaluation (epoch {epoch}/{num_epochs}): "
                ):
                    output = model(batch.to(device))
                    loss = criterion(output, batch.y)
                    total_loss += loss.item()
                    total_metrics.update(compute_metrics(output, batch.y, epoch))
                    all_predictions.append(output)
    else:
        for batch in tqdm(data, desc=f"Training (epoch {epoch}/{num_epochs}): "):
            optimizer.zero_grad()
            output = model(batch.to(device), training=True)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_metrics.update(compute_metrics(output, batch.y, epoch))
            all_predictions.extend(output)

    if output_pred and output_dire:
        predictions = np.concatenate(all_predictions, axis=0)
        pred_dir = output_dire + "/Preds"
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        curr_time = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        path_name = f"{pred_dir}/loads_{curr_time}.npy"
        np.save(path_name, predictions)

        print(f"\nLoads.npy file generated at : {path_name}")

    if not pred_only:
        avg_loss = total_loss / nb_batch
        avg_metrics = {
            metrics: scores / nb_batch for metrics, scores in total_metrics.items()
        }
        return avg_loss, avg_metrics

    return None, None

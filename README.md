# PowerGridSolverGNN

(The tasks are in the notebook : notebook_test.ipynb ; also check : basic_data_analysis.ipynb)

# README for Electricity Grid Load Forecasting with GNN

This project utilizes Graph Neural Networks (GNNs) for load forecasting in electricity grids. It includes scripts for training and evaluating the model, visualizing data, and a notebook for testing. The project is implemented using PyTorch and PyTorch Geometric.

## Dependencies
- Python 3.9+
- PyTorch (latest version)
- PyTorch Geometric (latest version)
- NumPy
- Matplotlib
- yaml
- argparse

## Files Description

### `dataset.py`
Handles data loading and preprocessing. It reads data from the specified directories, applies necessary transformations, and prepares it for the GNN model.
We use special features for the injections : (t: i - (lag_step * lag_jump), N) lag_step : [0, 4] and lag_jump : 10

### `model.py`
Contains the GNN model definition. This file defines the architecture of the GNN used for load forecasting, including layers, normalization, and any attention mechanisms.

### `train.py`
Script for training the GNN model. It reads configuration from a YAML file and trains the model on the provided dataset.
The training logs, checkpoints, config file and other are save in a logs folder : '/logs/{model}/... (that will be used for eval.py)

#### Arguments:
- `config_path`: Path to the configuration YAML file.
- `dir_path`: Path to the training dataset folder.
- `verbose`: Verbosity level for training output (integer).
- `save_plot`: Boolean flag to save training and validation loss plots.

### `eval.py`
Script for evaluating the trained GNN model. It uses the model checkpoint from training to make predictions on a validation or test dataset.
The prediction loads.npy file are generated in the '/logs/{model}/preds/'

#### Arguments:
- `config_path`: Path to the configuration YAML file used during training.
- `dir_path`: Path to the dataset folder for evaluation.
- `output_path`: Path where the predicted `loads.npy` will be saved.
- `verbose`: Verbosity level for evaluation output (integer).
- `pred_only`: Boolean flag to only save predictions without evaluating.

### `config.yaml`
A YAML file for model configuration. It includes parameters such as learning rate, number of epochs, model architecture details, etc.

### `notebook_test.ipynb`
Jupyter notebook for quick testing and experimenting with the model and data.

### `basic_data_analysis.ipynb`
Jupyter notebook for basic data visualization and analysis. Useful for understanding the dataset before training the model.

## Model and Training Details

- The model is trained using the data provided in the specified directory.
- Model checkpoints and configuration files are saved in the `/logs/models` directory.
- TensorBoard logs are generated for visualizing training progress.

## Visualization

- Loss plots and other relevant metrics during training and validation are saved if `save_plot` is set to `True`.
- The `basic_data_analysis.ipynb` notebook provides utilities for visualizing data distributions, correlations, and other relevant insights.

## Usage 

(see : 

1. **Training the Model**:
   ```
   python train.py --config_path=path/to/config.yaml --dir_path=path/to/dataset --verbose=2 --save_plot
   ```

2. **Evaluating the Model**:
   ```
   python eval.py --config_path=path/to/config.yaml --dir_path=path/to/validation_dataset --output_path=path/to/output --verbose=2 --pred_only
   ```

3. **Visualizing Data**:
   - Open `basic_data_analysis.ipynb` in Jupyter Notebook or JupyterLab for data visualization.
   - Use `notebook_test.ipynb` for quick model testing and experimentation.

## Notes

- Ensure all dependencies are installed and up-to-date.
- Adjust paths in the command-line arguments according to your directory structure.
- For detailed configuration, modify `config.yaml` as needed.

This README provides a comprehensive guide to navigating and utilizing the components of the project for effective load forecasting using GNNs.

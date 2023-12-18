# DeepLearningProject
This repository contains the final project of the DTU course 02456 Deep Learning Fall 2023. The project was carried out by Aimas Lund (s174435), Emma Christine Lei Hovmand (s194062) and Oda Byskov Aggerholm (s183700). 

As the project contains 14 trained models, it has not been possible to create a single notebook recreating the results of the work. Instead, seperate scripts for each model as well as data processing can be found.

This repositories contains five main folders:
1. `LSF_queries`: Containing a markdown description of how to use the HPC at DTU along with a template shell script for an HPC job.
2. `data_handling`: Containing the file `data_handler.py` with a datahandler class defined that can be used for fetching relevant data from the database it is initialized with.
3. `model_training`: Containing all 14 training scripts for the models.
4. `plot`: Containing all scripts used for data processing and plot generation.

## Requirements
The deep learning project in this repository is based on the `SchNetPack Toolbox` ([repository](https://github.com/atomistic-machine-learning/schnetpack) and [documentation](https://schnetpack.readthedocs.io/en/latest/)), and can be installed using the following command:

```bash
pip install schnetpack
pip install tensorboard
```

All models have been trained using `python 3.10.12`.

## 1. LSF Queries
`LSF_notes.md` describes how one can setup and run a training on the DTU HPC. `job_template.sh` is a template job script that requires few modifications to work to start a job on the HPC. A test model and corresponding job script is given in the folder `test_query` to serve as an example.

## 2. Data Handling
Along with the `QM9DataHandler` class, this folder contains all trained models in the folder `models` and a folder `data` that contains the validation data for each model in pickle file format.

The `QM9DataHandler` is initialized with qm9 data. It works like an iterable and returns all relevant data in `dict` format. The `dict` contains the following keys:

* `positions`: The positions of the atoms in the molecule.
* `atom_numbers`: The atomic numbers in sequence in the given molecule.
* `atom_mask`: A mask to indicate the positions of the chosen atom in the molecule.
* `properties`: The property values of the molecule.

When data is fetch, you can parse a trained model to the `fetch_model_output` function. This will return the output of the model for the given data. The output will update the previous dict, such that it will contain the following keys:

* `positions`: The positions of the atoms in the molecule.
* `atom_numbers`: The atomic numbers in sequence in the given molecule.
* `atom_mask`: A mask to indicate the positions of the chosen atom in the molecule.
* `properties`: The property values of the molecule.
* `embeddings`: The embeddings of the atoms in the molecule.
* `predictions`: The output of the model for the given molecule.

## 3. Model Training
The folder `training_scripts` contains a folder for each of the six models trained on individual properties and combined properties. The folder `training_scripts_retraining` contains a folder for each of the eight models trained for transfer learning (TL) and transfer learning with all except the output weights frozen (TLF).

Each folder for a given training contains the python script defining the model to train, the job script used to request the training on the HPC, and one (or two) result folders containing the trained model along with a folder called `lightning_logs`. The content of the `lightning_logs` displays graphs of training and validation losses etc. for the training. The logs can be displayed in a browser by running the prompt below:

```bash
tensorboard --logdir=qm9tut/lightning_logs
```

Here, `qm9tut` is the name of the result folder containing the `lightning_logs`.

## 4. Plot
The `plot` folder contains four subfolders with notebooks to generate all plots in the project. An overview is given below:

* `correlation_plot`: Containing a notebook with initial analysis of the data in the QM9 dataset. The generated plot is a correlation matrix of all properties in the dataset.
* `histograms`: Containing a notebook to generate the histogram plots.
* `loss_plots`: Containing a notebook to generate the plots with validation and training losses.
* `t-SNE`: Containing a notebook to generate the t-SNE plots.

Each of these folders also contains all generated plots for the given plot type.

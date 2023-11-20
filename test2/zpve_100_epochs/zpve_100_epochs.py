import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

# Make directory
qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

# Define constants
PROPERTY = QM9.zpve
EPOCHS = 100
BATCH_SIZE = 16

# Load data
split_file = ".qm9tut/split.npz"
if os.path.exists(split_file) :
    os.remove(split_file)

qm9data = QM9(
    './qm9.db', 
    batch_size=BATCH_SIZE,
    num_train=110000,
    num_val=10000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(PROPERTY, remove_mean=True, remove_atomrefs=True), #remove for homo and lumo
        trn.CastTo32()
    ],
    num_workers=2,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=True, # set to false, when not using a GPU
    load_properties=[PROPERTY] #only load relevant properties
)
qm9data.prepare_data()
qm9data.setup()

# Setup the model
cutoff = 5.
n_atom_basis = 32

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)
pred_property = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=PROPERTY)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_property],
    postprocessors=[trn.CastTo64(), trn.AddOffsets(PROPERTY, add_mean=True, add_atomrefs=True)] #remove AddOffsets for homo and lumo
)

output = spk.task.ModelOutput(
    name=PROPERTY,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

# Setup trainer and logger
logger = pl.loggers.TensorBoardLogger(save_dir=qm9tut)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(qm9tut, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=qm9tut,
    max_epochs=EPOCHS, # Set number of epochs
)
trainer.fit(task, datamodule=qm9data)
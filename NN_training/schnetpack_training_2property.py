import os
import schnetpack as spk
from schnetpack.datasets import QM9
import schnetpack.transform as trn

from schnetpack.data import ASEAtomsData
from schnetpack.transform import ASENeighborList

import torch
import torchmetrics
import pytorch_lightning as pl


#qm9tut = os.path.join(os.getcwd(), './qm9tut')
qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

################ Parameters #################
PROPERTY1 = QM9.homo
PROPERTY2 = QM9.lumo
BATCH_SIZE = 16
CUTOFF = 5.
LR = 1e-4
NUM_WORKERS = 4
NUM_ATOM_BASIS = 32
EPOCHS = 50
NUM_TRAIN = 110000
NUM_VALIDATION = 10000
T_ITERATIONS = 3

torch.manual_seed(0)

################ Load QM9 database ################
qm9data = QM9(
    #qm9db = os.path.join(os.getcwd(), '/qm9.db'),
    './qm9.db',
    batch_size=BATCH_SIZE,
    num_train=NUM_TRAIN,
    num_val=NUM_VALIDATION,
    transforms=[
        trn.ASENeighborList(cutoff=CUTOFF),
        # trn.RemoveOffsets(PROPERTY, remove_mean=True, remove_atomrefs=True),
        trn.CastTo32()
    ],
    #property_units={QM9.U0: 'eV'},
    num_workers=NUM_WORKERS,
    split_file=os.path.join(qm9tut, "split.npz"),
    pin_memory=False, # set to false, when not using a GPU
    load_properties=[PROPERTY1, PROPERTY2], # load relevant properties
)
qm9data.prepare_data()
qm9data.setup()

#################### model ####################
cutoff = CUTOFF
n_atom_basis = NUM_ATOM_BASIS

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=T_ITERATIONS,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_property1 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=PROPERTY1)
pred_property2 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=PROPERTY2)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_property1, pred_property2],
    # postprocessors=[trn.CastTo64(), trn.AddOffsets(PROPERTY, add_mean=True, add_atomrefs=True)]
    postprocessors=[trn.CastTo64()]
)

############## Output ##############
output_property1 = spk.task.ModelOutput(
    name=PROPERTY1,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_property2 = spk.task.ModelOutput(
    name=PROPERTY2,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

#################### task ####################
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_property1, output_property2],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": LR}
)
# NOTE: By default, the loss is summed over all outputs (see spk.task.AtomisticTask.loss_fn)

#################### training ####################
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
    max_epochs=EPOCHS, # for testing, we restrict the number of epochs
)
print('Start training')
trainer.fit(task, datamodule=qm9data)
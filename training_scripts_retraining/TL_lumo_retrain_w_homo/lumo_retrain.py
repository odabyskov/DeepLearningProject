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
qm9tut = './qm9tut_lumo'
if not os.path.exists('qm9tut_lumo'):
    os.makedirs(qm9tut)

################ Parameters #################
PROPERTY = QM9.homo
BATCH_SIZE = 16
CUTOFF = 5.
LR = 1e-4
NUM_WORKERS = 4
NUM_ATOM_BASIS = 32
EPOCHS = 50
NUM_TRAIN = 110000
NUM_VALIDATION = 10000
T_ITERATIONS = 3

PIN_MEMORY = True

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
    pin_memory=PIN_MEMORY, # set to false, when not using a GPU
    load_properties=[PROPERTY], # load relevant properties
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

pred_property = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=PROPERTY)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_property],
    # postprocessors=[trn.CastTo64(), trn.AddOffsets(PROPERTY, add_mean=True, add_atomrefs=True)]
    postprocessors=[trn.CastTo64()]
)

# Load weights from pre-trained model into a new nnpot model instance
# NOTE: Training directory on pretrained model will return errors
pre_trained_model = torch.load(os.path.join(qm9tut, "best_inference_model"))
nnpot.load_state_dict(pre_trained_model.state_dict())


############## Output ##############
output_property = spk.task.ModelOutput(
    name=PROPERTY,
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

#################### task ####################
task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_property],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": LR}
)


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
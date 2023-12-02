import torch
from tqdm import tqdm
from typing import List, Dict, Union
from schnetpack.datasets import QM9
import numpy as np
from ase import Atoms
import schnetpack as spk
import pickle
import os
import pickle


class QM9DataHandler:
    H = 1
    C = 6
    N = 7
    O = 8
    SAVE_DIR = os.path.join(os.getcwd(), "data")

    def __init__(self, qm9data: QM9, cutoff: float = 5.0):
        self.qm9data = qm9data
        self._batch_size = qm9data.batch_size

        if self._batch_size != 1:
            raise ValueError("Batch size of QM9 class must be 1.")

        self.converter = spk.interfaces.AtomsConverter(
            neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff),
            dtype=torch.float32,
        )
        self.properties = None
        self.property_values = None
        self.positions = None
        self.atom_numbers = None
        self.isolate_atom = None
        self.embeddings = None
        self.predictions = None

        self._POSITIONS_KEY = "positions"
        self._ATOM_NUMBERS_KEY = "atom_numbers"
        self._PROPERTIES_KEY = "properties"
        self._ATOM_MASK_KEY = "atom_mask"
        self._EMBEDDINGS_KEY = "embeddings"
        self._PREDICTIONS_KEY = "predictions"

    def __len__(self):
        return self.qm9data.num_val

    def __iter__(self) -> Dict[str, np.ndarray]:
        """
        Calls the generator function to iterate over the validation data.
        The validation data is transformed into the following format:
        {
            positions: np.ndarray,
            atom_numbers: np.ndarray,
            properties: np.ndarray
            atom_mask: np.ndarray (optional)
        }
        """
        if self.properties is None:
            raise ValueError("Properties not fetched yet.")
        for idx, p in enumerate(self.positions):
            tf_batch = {}
            tf_batch[self._POSITIONS_KEY] = p
            tf_batch[self._ATOM_NUMBERS_KEY] = self.atom_numbers[idx]
            tf_batch[self._PROPERTIES_KEY] = self.property_values[idx]

            if self.isolate_atom is not None:
                tf_batch[self._ATOM_MASK_KEY] = (
                    tf_batch[self._ATOM_NUMBERS_KEY] == self.isolate_atom
                )

            if self.embeddings is not None:
                tf_batch[self._EMBEDDINGS_KEY] = self.embeddings[idx]

            if self.predictions is not None:
                tf_batch[self._PREDICTIONS_KEY] = self.predictions[idx]

            yield tf_batch

    def fetch_data(self, properties: List[str]):
        """
        Fetches and transforms the validation data from the QM9 dataset.

        Args:
            properties (List[str]): List of properties to fetch.

        Raises:
            ValueError: If the requested properties are not in the QM9 dataset.
        """
        self.properties = None
        props = [None] * len(self)
        positions = [None] * len(self)
        atom_numbers = [None] * len(self)
        idx = 0

        with tqdm(total=len(self)) as pbar:
            for batch in self.qm9data.val_dataloader():
                # Check if properties are in QM9 dataset
                if self.properties is None:
                    if not set(properties).issubset(batch.keys()):
                        raise ValueError("Requested property not in dataset.")
                    else:
                        self.properties = properties

                tf_batch = self._transform_batch(batch)

                props[idx] = tf_batch[self._PROPERTIES_KEY]
                positions[idx] = tf_batch[self._POSITIONS_KEY]
                atom_numbers[idx] = tf_batch[self._ATOM_NUMBERS_KEY]

                idx += 1
                pbar.update(1)

            self.property_values = props
            self.positions = positions
            self.atom_numbers = atom_numbers

    def fetch_model_outputs(self, model: spk.model.NeuralNetworkPotential):
        """
        Fetches the embeddings for the validation data.

        Args:
            model (spk.model.NeuralNetworkPotential): Model to fetch embeddings from.
        """
        if self.positions is None:
            raise ValueError("Data is not fetched yet.")

        self.embeddings = None
        self.predictions = None

        embeddings = [None] * len(self)
        predictions = [None] * len(self)
        idx = 0

        with tqdm(total=len(self)) as pbar:
            for data in self:
                atoms = Atoms(
                    numbers=data[self._ATOM_NUMBERS_KEY],
                    positions=data[self._POSITIONS_KEY],
                )
                inputs = self.converter(atoms)
                pred, digest = self._get_embedding_digest(model, inputs)

                try:
                    predictions[idx] = np.array(
                        [pred[prop].item() for prop in self.properties]
                    )
                except KeyError:
                    raise RuntimeError(
                        "The provided model does not output the requested properties."
                    )

                embeddings[idx] = np.array(digest).squeeze()

                idx += 1
                pbar.update(1)

            self.embeddings = embeddings
            self.predictions = predictions

    def set_atom_isolation(self, atom_number: int):
        """
        Sets the filter for the validation data.

        Args:
            filtered_atom (int): Atom number to filter.
        """
        self.isolate_atom = atom_number

    def save_outputs(self, model_name: str = None):
        """
        Saves the acquired data to a pickle file.
        """
        data = np.array([None] * len(self))
        for idx, molecule in enumerate(self):
            data[idx] = molecule

        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        if model_name is not None:
            filename = f"outputs_{model_name}.pkl"
        else:
            filename = f"outputs.pkl"
        filepath = os.path.join(self.SAVE_DIR, filename)
        index = 1
        while os.path.exists(filepath):
            if model_name is not None:
                filename = f"outputs_{model_name}({index}).pkl"
            else:
                filename = f"outputs_{index}.pkl"
            filepath = os.path.join(self.SAVE_DIR, filename)
            index += 1

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def _get_embedding_digest(
        self,
        model: spk.model.NeuralNetworkPotential,
        inputs: spk.interfaces.AtomsConverter,
    ) -> Union[torch.Tensor, List[np.ndarray]]:
        digest = []

        def hook_callback(module, input: torch.Tensor, output):
            nonlocal digest
            digest.append(input[0].detach().numpy())

        model_layer = model.output_modules[0].outnet[0]
        hook_handle = model_layer.register_forward_hook(hook_callback)

        model.eval()
        pred = model(inputs)

        # in case of multiple outputs, we need to cast the tuple to a dict
        if isinstance(pred, tuple):
            pred = {k: v for k, v in pred}

        hook_handle.remove()
        return pred, digest

    def _transform_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Transforms the batch into a format easier to work with.
        NOTE: Assumes that the batch size is 1 and provided properties are valid.

        Args:
            batch (Dict[str, torch.Tensor]): Batch from QM9 dataset.

        Returns:
            Dict[str, np.ndarray]: Batch in numpy format with only the relevant properties.
        """
        tf_batch = {}
        tf_batch[self._POSITIONS_KEY] = batch["_positions"].cpu().numpy()
        tf_batch[self._ATOM_NUMBERS_KEY] = batch["_atomic_numbers"].cpu().numpy()
        tf_batch[self._PROPERTIES_KEY] = np.array(
            [batch[prop].item() for prop in self.properties]
        )

        return tf_batch

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch_geometric.loader import DataLoader as DataLoaderPyG

from .mol_dataset import QM9Dataset
from .spectre_dataset import SpectreGraphDataset
from .tokenizer import Graph2TrailTokenizer


def add_dataset_name(data, dataset_name):
    data.dataset_name = dataset_name
    return data


class GraphDataset(pl.LightningDataModule):
    datasets_map: ClassVar[dict[str, Callable]] = {
        "planar": partial(SpectreGraphDataset, dataset_name="planar"),
        "QM9": QM9Dataset,
    }

    def __init__(
        self,
        root,
        dataset_names="all",
        tokenizer=None,
        init_tokenizer=True,
        max_length=-1,
        truncation_length=None,
        labeled_graph=False,
        undirected=True,
        linearization_strategy="random_order",
        subset_size=None,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.dataset_names = dataset_names
        self.set_val_metric(dataset_names)
        if dataset_names == "all":
            self.dataset_names = list(self.datasets_map.keys())
        else:
            if not isinstance(dataset_names, list):
                self.dataset_names = [dataset_names]
            for dataset_name in self.dataset_names:
                assert dataset_name in self.datasets_map, "Not included in the database!"

        self.labeled_graph = labeled_graph
        self.subset_size = subset_size
        self.tokenizer = tokenizer
        if tokenizer is None and init_tokenizer:
            self.tokenizer = Graph2TrailTokenizer(
                # dataset_names=self.dataset_names,
                dataset_names=[],
                max_length=max_length,
                truncation_length=truncation_length,
                labeled_graph=labeled_graph,
                undirected=undirected,
                linearization_strategy=linearization_strategy,
            )

        self.kwargs = kwargs
        self.collate_fn = self.tokenizer.batch_converter() if self.tokenizer is not None else None

    def set_val_metric(self, dataset_names):
        if dataset_names == "planar":
            self.val_metric = ("vun_score", "max")
        else:
            self.val_metric = ("loss", "min")

    def _get_first_dataset(self):
        """Get the first dataset, handling Subset and ConcatDataset wrappers."""
        dataset = self.train_dataset
        # Unwrap Subset if present
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        # Get first dataset from ConcatDataset
        if hasattr(dataset, "datasets"):
            return dataset.datasets[0]
        return dataset

    @property
    def atom_decoder(self):
        if self.labeled_graph:
            # TODO: suppport multiple molecule datasets
            return self._get_first_dataset().atom_decoder
        return None

    @property
    def train_smiles(self):
        if self.labeled_graph:
            return self._get_first_dataset().smiles
        return None

    def prepare_data(self):
        max_num_nodes = {}
        for dataset_name in self.dataset_names:
            train_dataset = self.datasets_map[dataset_name](
                root=self.root,
                split="train",
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            )
            self.datasets_map[dataset_name](
                root=self.root,
                split="val",
                pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
            )

            # Check if test split should be processed
            # We only process if it's explicitly supported or likely to exist
            try:
                self.datasets_map[dataset_name](
                    root=self.root,
                    split="test",
                    pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                )
            except Exception as e:
                # Log but don't crash, as some datasets (like BA/ER in current config)
                # might only have train/val splits defined.
                print(f"⚠️ Skipping test split for {dataset_name}: {e}")

            if self.tokenizer is not None:
                if hasattr(train_dataset, "num_nodes") and torch.is_tensor(train_dataset.num_nodes):
                    max_num_nodes[dataset_name] = int(train_dataset.num_nodes.max())
                elif (
                    hasattr(train_dataset, "data")
                    and hasattr(train_dataset.data, "num_nodes")
                    and torch.is_tensor(train_dataset.data.num_nodes)
                ):
                    max_num_nodes[dataset_name] = int(train_dataset.data.num_nodes.max())
                else:
                    # Use generator expression instead of list comprehension for memory efficiency
                    max_num_nodes[dataset_name] = max(g.num_nodes for g in train_dataset)
        if self.tokenizer is not None:
            self.max_num_nodes = max_num_nodes
            self.tokenizer.set_num_nodes(max(max_num_nodes.values()))
            print(self.max_num_nodes)
            if self.labeled_graph:
                # TODO: support multiple molecule datasets
                self.tokenizer.set_num_node_and_edge_types(
                    num_node_types=train_dataset.num_node_types,
                    num_edge_types=train_dataset.num_edge_types,
                )
            else:
                self.tokenizer.set_num_node_and_edge_types()

    def setup(self, stage="fit"):
        if stage == "fit":
            train_dataset = [
                self.datasets_map[dataset_name](
                    root=self.root,
                    split="train",
                    transform=self.tokenizer,
                    pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                )
                for dataset_name in self.dataset_names
            ]
            train_dataset = ConcatDataset(train_dataset)
            if self.subset_size is not None:
                # Fixed seed for reproducibility
                generator = torch.Generator().manual_seed(42)
                indices = torch.randperm(len(train_dataset), generator=generator)[
                    : self.subset_size
                ].tolist()
                train_dataset = Subset(train_dataset, indices)
            self.train_dataset = train_dataset
            val_dataset = [
                self.datasets_map[dataset_name](
                    root=self.root,
                    split="val",
                    transform=self.tokenizer,
                    pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                )
                for dataset_name in self.dataset_names
            ]
            self.val_dataset = ConcatDataset(val_dataset)

        if stage == "test":
            test_dataset_list = []
            for dataset_name in self.dataset_names:
                try:
                    ds = self.datasets_map[dataset_name](
                        root=self.root,
                        split="test",
                        transform=self.tokenizer,
                        pre_transform=partial(add_dataset_name, dataset_name=dataset_name),
                    )
                    test_dataset_list.append(ds)
                except Exception as e:
                    print(f"⚠️ Could not load test split for {dataset_name}: {e}")

            if test_dataset_list:
                self.test_dataset = ConcatDataset(test_dataset_list)
            else:
                self.test_dataset = None

    def dataloader(self, dataset, **kwargs):
        if self.tokenizer is None:
            return DataLoaderPyG(dataset, **kwargs)
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.train_dataset, shuffle=True, collate_fn=self.collate_fn, **self.kwargs
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(
            self.val_dataset, shuffle=False, collate_fn=self.collate_fn, **self.kwargs
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(
            self.test_dataset, shuffle=False, collate_fn=self.collate_fn, **self.kwargs
        )

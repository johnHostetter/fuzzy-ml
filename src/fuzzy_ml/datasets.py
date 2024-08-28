"""
Implements classes or functions related to datasets.
"""

import csv
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Tuple

import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import sklearn.datasets
import pandas as pd
import numpy as np


@dataclass
class RegressionDatasetConfig:
    """
    A configuration for a regression dataset.
    """

    name: str
    frame: pd.DataFrame
    feature_names: List[str]
    target_names: List[str]


class RegressionImageFolder(datasets.ImageFolder):
    """
    A custom ImageFolder class that stores the targets as a tensor.
    """

    def __init__(
        self, root: str, image_scores: Dict[str, float], **kwargs: Any
    ) -> None:
        super().__init__(root, **kwargs)
        paths, _ = zip(*self.imgs)
        self.targets: torch.Tensor = torch.Tensor(
            [image_scores[path] for path in paths]
        ).cpu()
        self.samples = self.imgs = list(zip(paths, self.targets))


class LabeledDataset(Dataset):
    """
    A custom Dataset class that stores relates the inputs with their targets.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: Union[None, torch.Tensor] = None,
        out_features: Union[None, int] = None,
    ):
        """
        Initialize the LabeledDataset.

        Args:
            data: The input data.
            labels: The optional target data. If None, out_features must be provided.
            out_features: The number of output features. If None, labels must be provided.
        """
        self.data: torch.Tensor = data
        self.labels: Union[None, torch.Tensor] = labels
        if labels is None:
            if out_features is None:
                raise ValueError("If labels is None, out_features must be provided.")
            self.out_features = out_features
        else:
            if out_features is not None:
                raise ValueError("If labels is not None, out_features must be None.")
            self.out_features = labels.shape[-1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class RegressionDatasets:
    """
    The regression datasets, some are from OpenML and others are from RKeel.
    """

    def __init__(self):
        # create a dictionary of the datasets that are available in RKEEL (value is None if
        # available), else the dataset ID is given for OpenML
        self.lookup_table = {
            "autoMPG6": None,
            "delta_elv": None,
            "forestFires": None,
            "friedman": None,
            "mv": None,
            "plastic": None,
            "house_16H": 574,
            "Ailerons": 296,
            "treasury": 42367,
            "weather_izmir": 42369,
            "california_housing": 43939,
            "pumadyn32nh": 44981,
            "pol": 44133,
        }
        self.datasets = {}
        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        utils.install_packages("RKEEL")
        self.keel_package = importr("RKEEL")
        self.load_datasets()

    def __getitem__(self, item: str) -> RegressionDatasetConfig:
        """
        Get an item from the datasets.

        Args:
            item: The item to get.

        Returns:
            The dataset configuration.
        """
        return self.datasets[item]

    def load_datasets(self) -> None:
        """
        Load the datasets.

        Returns:
            None
        """
        for name, dataset_id in self.lookup_table.items():
            if dataset_id is not None:
                self.datasets[name] = self.load_openml_dataset(name, dataset_id)
            else:
                self.datasets[name] = self.load_rkeel_dataset(name)

    @staticmethod
    def load_openml_dataset(
        dataset_name: str, dataset_id: int
    ) -> RegressionDatasetConfig:
        """
        Load a dataset from OpenML.

        Args:
            dataset_name: The name of the dataset to load.
            dataset_id: The ID of the dataset to load.

        Returns:
            The dataset configuration.
        """
        # the dataset is not available in RKEEL, so must be downloaded from OpenML
        openml_results: dict = sklearn.datasets.fetch_openml(
            data_id=dataset_id, return_X_y=False, as_frame=True, parser="auto"
        )

        return RegressionDatasetConfig(
            name=dataset_name,
            frame=openml_results["frame"],
            feature_names=openml_results["feature_names"],
            target_names=openml_results["target_names"],
        )

    def load_rkeel_dataset(self, dataset_name) -> RegressionDatasetConfig:
        """
        Load a dataset from RKEEL.

        Args:
            dataset_name: The name of the dataset to load.

        Returns:
            The dataset configuration.
        """
        # the dataset is available in RKEEL
        dataset: rpy2.robjects.vectors.DataFrame = self.keel_package.loadKeelDataset(
            dataset_name
        )
        with (ro.default_converter + pandas2ri.converter).context():
            dataset_df: pd.DataFrame = ro.conversion.get_conversion().rpy2py(dataset)

        return RegressionDatasetConfig(
            name=dataset_name,
            frame=dataset_df,
            feature_names=list(dataset_df.columns)[:-1],
            target_names=list(dataset_df.columns)[-1],
        )


def convert_dat_to_csv(
    dat_file_name: Path,
) -> RegressionDatasetConfig:
    """
    Convert a .dat file to a .csv file.
    """
    csv_file_name = str(dat_file_name).replace(".dat", ".csv")
    with (
        open(dat_file_name, encoding="utf-8") as dat_file,
        open(csv_file_name, "w", encoding="utf-8") as csv_file,
    ):
        csv_writer = csv.writer(csv_file)
        input_columns, output_columns = [], []
        for line in dat_file:
            if "@relation" in line or "@attribute" in line:
                continue
            if "@inputs" in line:
                line = line.replace("@inputs", "")
                row = [field.strip() for field in line.split(",")]
                input_columns.extend(row)
            elif "@outputs" in line or "@output" in line:
                line = line.replace("@outputs", "")
                line = line.replace("@output", "")
                row = [field.strip() for field in line.split(",")]
                output_columns.extend(row)
            elif "@data" in line:
                csv_writer.writerow(input_columns + output_columns)
            else:
                row = [field.strip() for field in line.split(",")]
                csv_writer.writerow(row)
    return RegressionDatasetConfig(
        name=dat_file_name.stem,
        frame=pd.read_csv(csv_file_name),
        feature_names=input_columns,
        target_names=output_columns,
    )


def load_from_openml_and_split(
    data_id: int, random_state: int
) -> Tuple[LabeledDataset, LabeledDataset, LabeledDataset]:
    """
    Load the dataset from sklearn.datasets.fetch_openml and split it into train, val, and test sets.

    Args:
        data_id: The ID of the dataset to load from OpenML.
        random_state: The random state to use for reproducibility.

    Returns:
        train_df, val_df, test_df

    """
    dataset = sklearn.datasets.fetch_openml(
        data_id=data_id, return_X_y=False, as_frame=True, parser="auto"
    )
    if isinstance(dataset, dict):
        # pylint: disable=no-member
        print(dataset.data.shape)
        data_frame: pd.DataFrame = dataset.data
        data_frame["target"] = dataset.target.values
    else:
        raise ValueError("The dataset is not a dictionary.")
    # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
    # 60% to train, 20% to val, 20% to test
    # resolve the possible unbalanced tuple unpacking in the following code block
    # pylint: disable=unbalanced-tuple-unpacking
    train_df, val_df, test_df = np.split(
        data_frame.sample(frac=1, random_state=random_state),
        [int(0.6 * len(data_frame)), int(0.8 * len(data_frame))],
    )
    input_features, target_features = data_frame.columns[:-1], data_frame.columns[-1]

    return (
        convert_data_frame_to_supervised_dataset(
            train_df, input_features=input_features, target_features=target_features
        ),
        convert_data_frame_to_supervised_dataset(
            val_df, input_features=input_features, target_features=target_features
        ),
        convert_data_frame_to_supervised_dataset(
            test_df, input_features=input_features, target_features=target_features
        ),
    )


def convert_data_frame_to_supervised_dataset(
    data_frame: pd.DataFrame, input_features: List[str], target_features: List[str]
) -> LabeledDataset:
    """
    Make the data into a supervised dataset.

    Args:
        data_frame: The dataframe to convert to a torch tensor.
        input_features: The input features.
        target_features: The target features.

    Returns:
        A supervised dataset.
    """
    input_data = torch.Tensor(data_frame[input_features].values)
    output_data = torch.Tensor(data_frame[target_features].values)
    return LabeledDataset(data=input_data, labels=output_data.unsqueeze(dim=-1))

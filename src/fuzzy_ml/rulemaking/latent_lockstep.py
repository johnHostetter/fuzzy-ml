"""
Functions related to the fuzzy logic rule creation process outlined as the Latent Lockstep Method.
"""

from typing import List, Type

import torch
from regime import Node
from fuzzy.logic.rule import Rule
from fuzzy.logic.variables import LinguisticVariables
from fuzzy.relations.t_norm import TNorm
from skorch import NeuralNetRegressor

from crisp_ml.autoencode import AutoEncoder
from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.rulemaking.common import find_maximum_fuzzy_terms
from fuzzy_ml.partitioning.clip import CategoricalLearningInducedPartitioning as CLIP


class LatentSpace(Node):
    """
    This class is used to find the latent space representation of the input data, then apply CLIP
    to find the fuzzy sets in that latent space.
    """

    def __init__(self, resource_name: str = "latent_linguistic_variables"):
        super().__init__(resource_name=resource_name)

    # @hyperparameter(*CLIP.hyperparameters().keys())
    def __call__(
        self,
        train_dataset: LabeledDataset,
        autoencoder: NeuralNetRegressor,
        device: torch.device,
        epsilon: float,
        adjustment: float,
    ) -> LinguisticVariables:
        # epsilon & adjustment are hyperparameters that are passed to the CLIP algorithm
        """
        The Latent-Lockstep Method for High-Dimensional Fuzzy Logic Rule Generation
        (by John Wesley Hostetter).

        Find the input data's representation in the latent space, then apply CLIP, followed by the
        Latent Lockstep Method to produce the fuzzy logic rules.

        Args:
            train_dataset: The labeled training dataset.
            autoencoder: The auto-encoder to use (after training).
            device: The torch device to use.

        Returns:
            A list of fuzzy logic rules.
        """
        assert isinstance(autoencoder.module, AutoEncoder), (
            "The autoencoder must be an instance of AutoEncoder. "
            f"Received {type(autoencoder.module)}."
        )
        hidden_space = self.latent_space(train_dataset.data, autoencoder.module, device)
        if train_dataset.labels is None:
            return CLIP(resource_name=self.resource_name)(
                LabeledDataset(
                    data=hidden_space, out_features=train_dataset.out_features
                ),
                device,
                epsilon,
                adjustment,
            )
        return CLIP(resource_name=self.resource_name)(
            LabeledDataset(data=hidden_space, out_features=-1),
            device,
            epsilon,
            adjustment,
        )

    @staticmethod
    def latent_space(
        input_tensor: torch.Tensor,
        autoencoder: AutoEncoder,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Given the input data and its auto-encoder, convert the input data to its latent space
        representation.

        Args:
            input_tensor: The input data.
            autoencoder: The auto-encoder to use.
            device: The torch device to use.

        Returns:
            The latent space representation.
        """
        with torch.no_grad():
            return autoencoder.encoder(input_tensor.float().to(device=device))


class LatentLockstep(Node):
    """
    The Latent Lockstep Method for fuzzy logic rule generation.

        "Latent Space Encoding for Interpretable Fuzzy Logic Rules in Continuous
        and Noisy High-Dimensional Spaces" by John Wesley Hostetter & Min Chi, FUZZ IEEE 2023.
    """

    def __init__(self, t_norm: Type[TNorm], resource_name: str = "rules"):
        super().__init__(resource_name=resource_name)
        self.t_norm = t_norm

    def __call__(
        self,
        train_dataset: LabeledDataset,
        autoencoder: NeuralNetRegressor,
        linguistic_variables,
        latent_linguistic_variables,
    ) -> List[Rule]:
        """
        A helper method for the Latent-Lockstep Method. Given the input data, the auto-encoder,
        antecedents (in the original space), and a configuration, generate only the fuzzy logic
        rules that are unique in the latent space.

        Args:
            train_dataset: The labeled training dataset.
            linguistic_variables: The antecedents (in the original space).
            latent_linguistic_variables: The antecedents (in the latent space).

        Returns:
            A list of fuzzy logic rules.
        """
        assert isinstance(autoencoder.module, AutoEncoder), (
            "The autoencoder must be an instance of AutoEncoder. "
            f"Received {type(autoencoder.module)}."
        )
        rules, rules_in_latent_space = [], []
        for observation in train_dataset.data:
            # this block of code is for antecedents
            latent_rule, _ = find_maximum_fuzzy_terms(
                autoencoder.module.encoder(observation.to(device=autoencoder.device)),
                latent_linguistic_variables.inputs,
            )
            previous_size = len(rules_in_latent_space)
            rules_in_latent_space.append(latent_rule)
            current_size = len(rules_in_latent_space)
            if current_size > previous_size:
                premise, _ = find_maximum_fuzzy_terms(
                    observation, linguistic_variables.inputs
                )
                # TODO: eliminate assumption of wanting TSK rules
                rules.append(
                    Rule(
                        premise=self.t_norm(*premise, device=autoencoder.device),
                        consequence=self.t_norm(
                            *{
                                (len(rules), i)
                                for i in range(train_dataset.out_features)
                            },
                            device=autoencoder.device,
                        ),
                    )
                )

        return rules

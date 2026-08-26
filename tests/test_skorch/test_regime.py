"""
White-box tests for fuzzy_ml.skorch.regime.SupervisedTraining.
"""

import unittest

import torch
from skorch.regressor import NeuralNetRegressor

from fuzzy_ml.datasets import LabeledDataset
from fuzzy_ml.skorch.regime import SupervisedTraining


class TestSupervisedTraining(unittest.TestCase):
    """
    Covers SupervisedTraining's hyperparameter defaults/overrides and its
    __call__ actually training a real (tiny) model end to end.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        data = torch.rand(20, 3)
        labels = torch.rand(20, 1)
        self.train_dataset = LabeledDataset(data=data, labels=labels)
        self.val_dataset = LabeledDataset(data=data, labels=labels)

    def test_default_hyperparameters(self) -> None:
        """
        Defaults documented in _setup_hyperparameters() are actually applied
        when no overrides are given.
        """
        training = SupervisedTraining()
        self.assertEqual(training.learning_rate, 1e-4)
        self.assertEqual(training.max_epochs, 12)
        self.assertEqual(training.batch_size, 32)
        self.assertEqual(training.patience, 4)
        self.assertEqual(training.monitor, "valid_loss")

    def test_hyperparameter_overrides(self) -> None:
        """
        Every hyperparameter can be overridden via keyword arguments at
        construction time.
        """
        training = SupervisedTraining(
            learning_rate=1e-2,
            max_epochs=3,
            batch_size=8,
            patience=1,
            monitor="train_loss",
        )
        self.assertEqual(training.learning_rate, 1e-2)
        self.assertEqual(training.max_epochs, 3)
        self.assertEqual(training.batch_size, 8)
        self.assertEqual(training.patience, 1)
        self.assertEqual(training.monitor, "train_loss")

    def test_resource_name_defaults_to_model(self) -> None:
        """
        SupervisedTraining is a regime.Node - its resource_name (used to wire
        it into a Regime graph) defaults to "model".
        """
        self.assertEqual(SupervisedTraining().resource_name, "model")

    def test_call_trains_a_real_model_and_returns_a_neural_net_regressor(
        self,
    ) -> None:
        """
        __call__ actually fits a real (tiny) PyTorch model via skorch and
        returns the fitted NeuralNetRegressor, with the original model
        accessible (and having been mutated in place by training) via
        .module_.
        """
        model = torch.nn.Linear(3, 1)
        training = SupervisedTraining(
            learning_rate=1e-2, max_epochs=2, batch_size=8, patience=1
        )

        result = training(model, self.train_dataset, self.val_dataset, self.device)

        self.assertIsInstance(result, NeuralNetRegressor)
        self.assertTrue(hasattr(result, "module_"))
        self.assertIs(result.module_, model)
        # confirm training actually ran (skorch records per-epoch history)
        self.assertGreaterEqual(len(result.history), 1)
        self.assertIn("train_loss", result.history[-1])
        self.assertIn("valid_loss", result.history[-1])


if __name__ == "__main__":
    unittest.main()

"""
Implements a generic auto-encoder that is used in the Latent-Lockstep Method.
"""

import torch


class AutoEncoder(torch.nn.Module):
    """
    A default auto-encoder; used for the latent-lockstep method.
    """

    def __init__(self, n_inputs: int, n_latent: int, device: torch.device):
        super().__init__()

        n_hidden: int = int(n_inputs / 2)
        self.encoder: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, n_hidden, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_latent, device=device),
            torch.nn.Tanh(),
        )
        self.decoder: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(n_latent, n_hidden, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(n_hidden, n_inputs, device=device),
            torch.nn.Tanh(),
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Encode and decode the given input (input_data), and then return
        the decoded representation.

        Args:
            input_data: The input data to encode and decode.

        Returns:
            The decoded representation of the input data.
        """
        encoded = self.encoder(input_data)
        decoded = self.decoder(encoded)
        return decoded.float()

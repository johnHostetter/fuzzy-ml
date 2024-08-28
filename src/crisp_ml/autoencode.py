"""
Implements a generic auto-encoder that is used in the Latent-Lockstep Method.
"""

import torch


class AutoEncoder(torch.nn.Module):
    """
    A default auto-encoder; used for the latent-lockstep method.
    """

    def __init__(self, in_features, latent_space_dim, device: torch.device):
        super().__init__()

        hidden_layer_size = int(in_features / 2)
        self.encoder: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_layer_size, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layer_size, latent_space_dim, device=device),
            torch.nn.Tanh(),
        )
        self.decoder: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(latent_space_dim, hidden_layer_size, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_layer_size, in_features, device=device),
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

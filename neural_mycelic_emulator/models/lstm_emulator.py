from __future__ import annotations

import torch
from torch import nn

class LSTMEmulator(nn.Module):
    """Simple causal LSTM for glyph prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: tuple | None = None):
        # tokens: (B, T)
        x = self.embed(tokens)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        logits = self.fc(out)  # (B, T, V)
        return logits, hidden

    def generate(self, start_token: int, max_len: int = 64, temperature: float = 1.0, device="cpu"):
        self.eval()
        token = torch.tensor([[start_token]], device=device)
        hidden = None
        seq = [start_token]
        with torch.no_grad():
            for _ in range(max_len - 1):
                logits, hidden = self(token, hidden)
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                seq.append(int(next_token.item()))
                token = next_token
        return seq 
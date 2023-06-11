import torch

class Attention(torch.nn.Module):
    def __init__(self, embed_dim, kdim=None, vdim=None):
        super(Attention, self).__init__()
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.embed_dim = embed_dim

        self.V = torch.nn.Linear(self.embed_dim, self.vdim)
        self.K = torch.nn.Linear(self.embed_dim, self.kdim)
        self.Q = torch.nn.Linear(self.embed_dim, self.kdim)

        self.fc_out = torch.nn.Linear(self.vdim, self.vdim)

    def forward(self, query, key, value):
        # Apply the Linear transformation
        v = self.V(value)
        k = self.K(key)
        q_scaled = self.Q(query) / (self.embed_dim ** (1 / 2))

        # Matrix multiplication of keys and queries: The core of attention mechanism
        energy = torch.einsum("nqd,nkd->nqk", [q_scaled, k])

        # Normalize energy values similarly to "softmax" so that they sum to 1.
        attention = torch.softmax(energy, dim=2)

        # Multiply attention values with the values tensor for the final self-attention output
        out = torch.einsum("nql,nld->nqd", [attention, v])

        out = self.fc_out(out)
        return out, attention
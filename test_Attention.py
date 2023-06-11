import unittest
import torch
from Attention import Attention

class TestAttention(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-5  # You may need to adjust this based on your application.

    def test_torch_compare(self):
        embed_dim = 1
        num_heads = 1
        batch_size = 1
        seq_len = 1

        # Initialize MultiheadAttention layer
        torch_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        # Ensure our SelfAttention and MultiheadAttention have the same initial weights
        attn = Attention(embed_dim)
        attn.V.weight.data = torch_attn.in_proj_weight.data[0:embed_dim].clone()
        attn.K.weight.data = torch_attn.in_proj_weight.data[embed_dim:(2*embed_dim)].clone()
        attn.Q.weight.data = torch_attn.in_proj_weight.data[(2*embed_dim):(3*embed_dim)].clone()
        attn.fc_out.weight.data = torch_attn.out_proj.weight.data.clone()

        # Share biases
        attn.Q.bias.data = torch_attn.in_proj_bias.data[0:embed_dim].clone()
        attn.K.bias.data = torch_attn.in_proj_bias.data[embed_dim:(2*embed_dim)].clone()
        attn.V.bias.data = torch_attn.in_proj_bias.data[(2*embed_dim):(3*embed_dim)].clone()
        attn.fc_out.bias.data = torch_attn.out_proj.bias.data.clone()

        # Example tensors
        query = torch.rand((batch_size, seq_len, embed_dim))
        key = torch.rand((batch_size, seq_len, embed_dim))
        value = torch.rand((batch_size, seq_len, embed_dim))

        # Apply attention
        torch_output, torch_attn_weights = torch_attn(query.transpose(0,1), key.transpose(0,1), value.transpose(0,1))
        output, attn_weights = attn(value, key, query)

        # Compare outputs
        self.assertTrue(torch.allclose(output, torch_output.transpose(0,1), atol=self.tolerance),
                        "The outputs are not close enough.")
        self.assertTrue(torch.allclose(attn_weights, torch_attn_weights, atol=self.tolerance),
                        "The attention weights are not close enough.")


if __name__ == '__main__':
    unittest.main()

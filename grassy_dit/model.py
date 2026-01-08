"""
GRASSY-DiT: GraphDiT with scattering moment conditioning.
Extends torch-molecule by adding:
1. ScatteringTokenizer: 440-D → [B, 21, D] dual tokens (atom + level)
2. CrossAttention: graph attends to scattering
3. SELayerWithCrossAttention: self-attn + cross-attn + MLP

Scattering structure (from GRASSY):
  440-D = A atom types × L levels × M moments = 10 × 11 × 4
  
  Levels (11): 1 zeroth + 4 first-order + 6 second-order
  Moments (4): mean, variance, skew, kurtosis
  
Dual tokenization:
  - Atom tokens [10]: each = 44 dims (11 levels × 4 moments)
  - Level tokens [11]: each = 40 dims (10 atoms × 4 moments)
  - Same data, two views → overlap enables flexible querying
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_molecule.generator.graph_dit.transformer import AttentionWithNodeMask as Attention, MLP as Mlp, TimestepEmbedder, FinalLayer as OutLayer


class CrossAttention(nn.Module):
    """Q from graph, K/V from scattering tokens."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias) # graph tokens -> query
        self.k = nn.Linear(dim, dim, bias=qkv_bias) # scattering tokens -> key
        self.v = nn.Linear(dim, dim, bias=qkv_bias) # scattering tokens -> value
        self.proj = nn.Linear(dim, dim) # output projection

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, context):
        B, N, D = x.shape
        K = context.shape[1]
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        out = F.scaled_dot_product_attention(q, k, v) 
        return self.proj(out.transpose(1, 2).reshape(B, N, D)) 


class ScatteringTokenizer(nn.Module):
    """
    Dual tokenization: 440-D scattering → [B, num_atom_types + num_levels, D]
    
    Atom tokens: "What's each atom type's full scattering signature?"
    Level tokens: "What's happening at each scattering order/scale?"
    
    Same moment data appears in both → model can query by atom OR by level.
    """
    
    def __init__(self, hidden_size=384, num_atom_types=16, num_levels=11, 
                 num_moments=4, dropout=0.1): # should change the default to be the actual number of atom types 
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_levels = num_levels
        self.num_moments = num_moments
        self.dropout = dropout
    
        # Total tokens = atom tokens + level tokens
        self.num_tokens = num_atom_types + num_levels  # the double tokenization method discussed
        
        # prjecting the per atom tokens
        self.atom_proj = nn.Linear(num_levels * num_moments, hidden_size)
        
        # projecting the per level tokens
        self.level_proj = nn.Linear(num_atom_types * num_moments, hidden_size)
        
        # Positional embeddings for all tokens
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, hidden_size) * 0.02)
        
        # Null embedding for CFG
        self.null = nn.Parameter(torch.randn(1, self.num_tokens, hidden_size) * 0.02)
        

    def forward(self, x, train=False, force_null=False):
        """
        Args:
            x: [B, 440] scattering moments
            train: whether in training mode (enables dropout)
            force_null: return null tokens (for unconditional generation)
        Returns:
            tokens: [B, 21, D] dual scattering tokens
        """
        B = x.shape[0]
        
        if force_null:
            return self.null.expand(B, -1, -1)
        
        # Reshape: [B, 440] → [B, A, L, M] = [B, 10, 11, 4] -- assuming this order of extraction of moments from the GRASSY scatter model
        x = x.view(B, self.num_atom_types, self.num_levels, self.num_moments)
        
        # Atom tokens: [B, A, L*M] = [B, 10, 44] → [B, 10, D]
        atom_tokens = x.view(B, self.num_atom_types, -1)
        atom_tokens = self.atom_proj(atom_tokens)
        
        # Level tokens: [B, L, A*M] = [B, 11, 40] → [B, 11, D]
        level_tokens = x.permute(0, 2, 1, 3).reshape(B, self.num_levels, -1)
        level_tokens = self.level_proj(level_tokens)
        
        # Concat: [B, A+L, D] = [B, 21, D]
        tokens = torch.cat([atom_tokens, level_tokens], dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.pos
        
        # CFG dropout during training
        if train and self.dropout > 0:
            mask = torch.rand(B, device=x.device) < self.dropout
            if mask.any():
                tokens[mask] = self.null.expand(mask.sum(), -1, -1)
        
        return tokens


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SELayerWithCrossAttention(nn.Module):
    """DiT block: self-attn + cross-attn + MLP, all modulated by timestep (AdalN)."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_head=num_heads, qkv_bias=True, qk_norm=True)
        
        # Cross-attention to scattering
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias=True, qk_norm=True)
        
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), use_bn=False) 
        
        # AdaLN modulation from timestep: 9 params (shift_i, scale_i, gate_i for i=1,2,3) 
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size)
        )

    def forward(self, x, c, node_mask, scatter_tokens):
        # c: [B, D] timestep embedding
        shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = self.adaLN(c).chunk(9, dim=1)
        
        # Self-attention (modulated)
        h = modulate(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.attn(h, node_mask=node_mask)
        
        # Cross-attention to scattering (modulated)
        h = modulate(self.norm_cross(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.cross_attn(h, scatter_tokens)
        
        # MLP (modulated)
        h = modulate(self.norm2(x), shift3, scale3)
        x = x + gate3.unsqueeze(1) * self.mlp(h)
        
        return x


class ScatteringDenoiser(nn.Module):
    """
    GraphDiT conditioned on GRASSY scattering moments.
    
    Args:
        max_n_nodes: maximum atoms per molecule (N)
        hidden_size: transformer hidden dimension (D)
        depth: number of transformer blocks
        num_heads: attention heads
        Xdim: atom types (A)
        Edim: bond types (5)
        num_atom_types: atom types in scattering (10)
        num_levels: scattering levels (11)
        num_moments: statistical moments (4)
    """
    
    def __init__(self, max_n_nodes, hidden_size=384, depth=12, num_heads=16,
                 mlp_ratio=4.0, Xdim=10, Edim=5, 
                 num_atom_types=16, num_levels=11, num_moments=4):
        super().__init__()
        self.max_n_nodes = max_n_nodes
        self.hidden_size = hidden_size
        
        # Input embeddings
        self.x_embedder = nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.scatter_tokenizer = ScatteringTokenizer(
            hidden_size=hidden_size,
            num_atom_types=num_atom_types,
            num_levels=num_levels,
            num_moments=num_moments
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SELayerWithCrossAttention(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Output projection - Imported from torch-molecule (FinalLayer). Projects transformer output to atom/bond predictions.
        self.out_layer = OutLayer(max_n_nodes, hidden_size, Xdim, Edim, mlp_ratio, num_heads)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init AdaLN output for stable training
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)

    def forward(self, x, e, node_mask, t, scattering, uncond=False):
        """
        Args:
            x: [B, N, Xdim] atom types (one-hot or noised)
            e: [B, N, N, Edim] bond types (one-hot or noised)
            node_mask: [B, N] valid atoms mask
            t: [B] diffusion timestep
            scattering: [B, 440] GRASSY scattering moments
            uncond: if True, use null tokens (for CFG)
        Returns:
            x_pred: [B, N, Xdim] predicted atom logits
            e_pred: [B, N, N, Edim] predicted bond logits
        """
        B, N = x.shape[:2]
        x_in, e_in = x, e
        
        # Embed inputs
        x = self.x_embedder(torch.cat([x, e.reshape(B, N, -1)], dim=-1))
        c = self.t_embedder(t)
        scatter_tokens = self.scatter_tokenizer(scattering, self.training, uncond)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, node_mask, scatter_tokens)
        
        # Output projection
        X_pred, E_pred, _ = self.out_layer(x, x_in, e_in, c, t, node_mask) 
        return X_pred, E_pred       

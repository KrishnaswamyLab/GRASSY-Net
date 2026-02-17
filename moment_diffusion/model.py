"""
GRASSY-DiT: GraphDiT with scattering moment conditioning.
Extends torch-molecule by adding:
1. ScatteringTokenizer: 440-D → [B, num_tokens, D] (atom + level + moment tokens)
2. CrossAttention: graph attends to scattering (with optional bottleneck)
3. SELayerWithCrossAttention: self-attn + cross-attn + MLP

Scattering structure (from GRASSY):
  440-D = A atom types × L levels × M moments = 10 × 11 × 4
  
  Levels (11): 1 zeroth + 4 first-order + 6 second-order
  Moments (4): mean, variance, skew, kurtosis
  
Tokenization options:
  - Atom tokens [A]: each = L*M dims → D via projection
  - Level tokens [L] (optional): each = A*M dims → D via projection
  - Moment tokens [M] (optional): each = A*L dims → D via projection
  
Projection options:
  - use_fixed_projections=False: learned nn.Linear (default)
  - use_fixed_projections=True: fixed orthogonal matrices (prevents overfitting)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_molecule.generator.graph_dit.transformer import AttentionWithNodeMask as Attention, MLP as Mlp, TimestepEmbedder, FinalLayer as OutLayer


class CrossAttention(nn.Module):
    """Q from graph, K/V from scattering tokens.
    
    Optional bottleneck: project to smaller dim before attention to reduce
    overfitting to conditioning signal while preserving base model capacity.
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0, bottleneck_dim=None):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.qk_norm = qk_norm
        
        # Bottleneck projections (optional) - reduces conditioning capacity
        if bottleneck_dim is not None:
            self.q_down = nn.Linear(dim, bottleneck_dim)
            self.kv_down = nn.Linear(dim, bottleneck_dim)
            self.out_up = nn.Linear(bottleneck_dim, dim)
            attn_dim = bottleneck_dim
        else:
            attn_dim = dim
        
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        
        # Q/K/V in attention space (bottleneck or full)
        self.q = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.k = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.v = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.proj = nn.Linear(attn_dim, attn_dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, context):
        B, N, _ = x.shape
        K = context.shape[1]
        
        # Bottleneck: project down to smaller space
        if self.bottleneck_dim is not None:
            x = self.q_down(x)
            context = self.kv_down(context)
        
        attn_dim = self.bottleneck_dim if self.bottleneck_dim else self.dim
        
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Manual attention with dropout (can't use F.scaled_dot_product_attention with dropout)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        
        out = self.proj(out.transpose(1, 2).reshape(B, N, attn_dim))
        
        # Bottleneck: project back up to original space
        if self.bottleneck_dim is not None:
            out = self.out_up(out)
        
        return self.proj_drop(out) 


class ScatteringTokenizer(nn.Module):
    """
    Dual/Triple tokenization: 440-D scattering → [B, num_tokens, D]
    
    Supports fixed orthogonal projections (non-learnable) to prevent overfitting.
    Only positional embeddings and null embeddings are learned when use_fixed_projections=True.
    
    Atom tokens: "What's each atom type's full scattering signature?"
    Level tokens: "What's happening at each scattering order/scale?"
    Moment tokens (optional): "What's the distribution shape across all atoms/levels?"
    
    Same moment data appears in multiple views → model can query by atom, level, or moment.
    """
    
    def __init__(self, hidden_size=384, num_atom_types=16, num_levels=11, 
             num_moments=4, dropout=0.1, moment_noise_cfg=None,
             use_moment_tokens=False, use_level_tokens=True, use_fixed_projections=False): 
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_levels = num_levels
        self.num_moments = num_moments
        self.dropout = dropout
        self.moment_noise_cfg = moment_noise_cfg or {}
        self.use_moment_tokens = use_moment_tokens
        self.use_level_tokens = use_level_tokens
        self.use_fixed_projections = use_fixed_projections
    
        # Total tokens = atom tokens + level tokens (optional) + moment tokens (optional)
        self.num_tokens = num_atom_types
        if use_level_tokens:
            self.num_tokens += num_levels
        if use_moment_tokens:
            self.num_tokens += num_moments
        
        # Projections: fixed orthogonal (non-learnable) or learned linear
        if use_fixed_projections:
            self.register_buffer('atom_proj', self._make_orthogonal_proj(num_levels * num_moments, hidden_size))
            if use_level_tokens:
                self.register_buffer('level_proj', self._make_orthogonal_proj(num_atom_types * num_moments, hidden_size))
            if use_moment_tokens:
                self.register_buffer('moment_proj', self._make_orthogonal_proj(num_atom_types * num_levels, hidden_size))
        else:
            self.atom_proj = nn.Linear(num_levels * num_moments, hidden_size)
            if use_level_tokens:
                self.level_proj = nn.Linear(num_atom_types * num_moments, hidden_size)
            if use_moment_tokens:
                self.moment_proj = nn.Linear(num_atom_types * num_levels, hidden_size)
        
        # Positional embeddings for all tokens (learned)
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, hidden_size) * 0.02)
        
        # Null embedding for CFG (learned)
        self.null = nn.Parameter(torch.randn(1, self.num_tokens, hidden_size) * 0.02)
    
    def _make_orthogonal_proj(self, in_dim, out_dim):
        """Create fixed orthogonal projection matrix [in_dim, out_dim] via QR decomposition."""
        random = torch.randn(out_dim, in_dim)
        q, _ = torch.linalg.qr(random)  # q: [out_dim, in_dim] with orthonormal columns
        return q.T  # [in_dim, out_dim]

    def _apply_moment_noise(self, x):
        """Apply moment noise augmentation during training."""
        cfg = self.moment_noise_cfg
        prob = cfg.get('prob', 0.0)
        if prob <= 0:
            return x
        
        B, total = x.shape
        device = x.device
        mask = torch.rand(B, device=device) < prob
        if not mask.any():
            return x
        

        noisy = x.clone()
        lower = max(1, int(cfg.get('lower_scalar', 0.25) * total))
        upper = int(cfg.get('upper_scalar', 1.0) * total)
        noise_std = cfg.get('noise_std', 0.1)
        
        for i in range(B):
            if mask[i]:
                q = torch.randint(lower, upper + 1, (1,), device=device).item()
                idx = torch.randperm(total, device=device)[:q]
                noisy[i, idx] += torch.randn(q, device=device) * noise_std
        return noisy    
        

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


        # Apply moment noise during training
        if train and self.moment_noise_cfg:
            x = self._apply_moment_noise(x)
        
        # Reshape: [B, 440] → [B, A, L, M] = [B, 10, 11, 4] -- assuming this order of extraction of moments from the GRASSY scatter model
        x = x.view(B, self.num_atom_types, self.num_levels, self.num_moments)
        
        # Atom tokens: [B, A, L*M] = [B, 10, 44] → [B, 10, D]
        atom_tokens = x.view(B, self.num_atom_types, -1)
        if self.use_fixed_projections:
            atom_tokens = atom_tokens @ self.atom_proj
        else:
            atom_tokens = self.atom_proj(atom_tokens)
        
        token_list = [atom_tokens]
        
        # Level tokens (optional): [B, L, A*M] = [B, 11, 40] → [B, 11, D]
        if self.use_level_tokens:
            level_tokens = x.permute(0, 2, 1, 3).reshape(B, self.num_levels, -1)
            if self.use_fixed_projections:
                level_tokens = level_tokens @ self.level_proj
            else:
                level_tokens = self.level_proj(level_tokens)
            token_list.append(level_tokens)
        
        # Moment tokens (optional): [B, M, A*L] = [B, 4, 110] → [B, 4, D]
        if self.use_moment_tokens:
            moment_tokens = x.permute(0, 3, 1, 2).reshape(B, self.num_moments, -1)
            if self.use_fixed_projections:
                moment_tokens = moment_tokens @ self.moment_proj
            else:
                moment_tokens = self.moment_proj(moment_tokens)
            token_list.append(moment_tokens)
        
        # Concatenate all tokens
        tokens = torch.cat(token_list, dim=1)
        
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
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cross_attn_drop=0.0,
                 cross_attn_bottleneck=None):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_head=num_heads, qkv_bias=True, qk_norm=True)
        
        # Cross-attention to scattering (with configurable dropout and optional bottleneck)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.cross_attn = CrossAttention(hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                                         attn_drop=cross_attn_drop, proj_drop=cross_attn_drop,
                                         bottleneck_dim=cross_attn_bottleneck)
        
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
                 num_atom_types=16, num_levels=11, num_moments=4, device=None,
                 moment_noise_cfg=None, use_moment_tokens=False, use_level_tokens=True,
                 use_fixed_projections=False,
                 cross_attn_drop=0.0, cross_attn_bottleneck=None):
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {device}")
        self.device = device
        self.max_n_nodes = max_n_nodes
        self.hidden_size = hidden_size
        self.use_moment_tokens = use_moment_tokens
        self.use_level_tokens = use_level_tokens
        self.use_fixed_projections = use_fixed_projections
        self.cross_attn_drop = cross_attn_drop
        self.cross_attn_bottleneck = cross_attn_bottleneck
        
        # Input embeddings
        self.x_embedder = nn.Linear(Xdim + max_n_nodes * Edim, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.scatter_tokenizer = ScatteringTokenizer(
            hidden_size=hidden_size,
            num_atom_types=num_atom_types,
            num_levels=num_levels,
            num_moments=num_moments,
            moment_noise_cfg=moment_noise_cfg,
            use_moment_tokens=use_moment_tokens,
            use_level_tokens=use_level_tokens,
            use_fixed_projections=use_fixed_projections,
        )
        
        # Transformer blocks (with cross-attention dropout and optional bottleneck)
        self.blocks = nn.ModuleList([
            SELayerWithCrossAttention(hidden_size, num_heads, mlp_ratio, 
                                      cross_attn_drop=cross_attn_drop,
                                      cross_attn_bottleneck=cross_attn_bottleneck)
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

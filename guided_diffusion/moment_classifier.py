"""
Moment Classifier: Predicts scattering moments from noisy molecular graphs.

This GNN regressor learns to predict what scattering moments the final clean
molecule will have, given a noisy graph at timestep t. The gradients from
this classifier can be used to guide generation toward target moments.

Architecture:
    1. Input embedding: X + E.flatten -> hidden_size
    2. Timestep embedding: sinusoidal -> MLP -> hidden_size
    3. Graph transformer layers (self-attention + timestep modulation)
    4. Global pooling: masked mean over nodes
    5. MLP head: hidden_size -> moment_dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by MLP projection.
    
    Similar to positional encoding in transformers, but for diffusion timesteps.
    """
    
    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        
        # MLP to project sinusoidal embedding to hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] or [B, 1] timestep values (can be int or float)
        
        Returns:
            [B, hidden_size] timestep embeddings
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
        
        t = t.float()
        device = t.device
        
        half_dim = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Pad if hidden_size is odd
        if self.hidden_size % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        
        return self.mlp(embedding)


class GraphSelfAttention(nn.Module):
    """
    Self-attention over graph nodes with node mask support.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, 
                 qkv_bias: bool = True, attn_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Layer norms for Q and K
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
    
    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] node features
            node_mask: [B, N] boolean mask for valid nodes
        
        Returns:
            [B, N, D] attended features
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply layer norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Attention scores
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]
        
        # Apply mask: invalid nodes shouldn't attend or be attended to
        mask_2d = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = attn.masked_fill(~mask_2d, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Handle all-masked rows (set to zero instead of nan)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Apply attention
        out = attn @ v  # [B, H, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)
        
        return self.proj(out)


class MLP(nn.Module):
    """Simple MLP block."""
    
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-style modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ClassifierBlock(nn.Module):
    """
    Transformer block for the classifier with timestep modulation.
    
    Structure: LayerNorm -> Self-Attention -> LayerNorm -> MLP
    With AdaLN modulation from timestep embedding.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, 
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = GraphSelfAttention(hidden_size, num_heads, attn_drop=dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = MLP(hidden_size, mlp_ratio, dropout)
        
        # AdaLN modulation: 6 values (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, 
                node_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] node features
            t_emb: [B, D] timestep embedding
            node_mask: [B, N] valid node mask
        
        Returns:
            [B, N, D] output features
        """
        # Get modulation parameters from timestep
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(t_emb).chunk(6, dim=1)
        
        # Self-attention with modulation
        h = modulate(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.attn(h, node_mask)
        
        # MLP with modulation
        h = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        
        return x


class MomentClassifier(nn.Module):
    """
    GNN classifier that predicts scattering moments from noisy molecular graphs.
    
    Given a noisy graph (X_t, E_t) at timestep t, predicts the scattering moments
    of the clean molecule. Used for classifier guidance during generation.
    
    Args:
        max_n_nodes: Maximum number of nodes in graph
        hidden_size: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        Xdim: Number of atom types
        Edim: Number of edge/bond types
        moment_dim: Output dimension (scattering moment size)
        mlp_ratio: MLP hidden size multiplier
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        max_n_nodes: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        Xdim: int = 10,
        Edim: int = 5,
        moment_dim: int = 440,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.max_n_nodes = max_n_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Xdim = Xdim
        self.Edim = Edim
        self.moment_dim = moment_dim
        
        # Input embedding: X + flattened E -> hidden_size
        input_dim = Xdim + max_n_nodes * Edim
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Timestep embedding
        self.t_embedder = SinusoidalTimestepEmbedding(hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ClassifierBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Output MLP: pooled features -> moment prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, moment_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-init AdaLN outputs for stable training
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
    
    def forward(
        self,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scattering moments from noisy graph.
        
        Args:
            X_t: [B, N, Xdim] noisy atom features (one-hot or soft)
            E_t: [B, N, N, Edim] noisy edge features (one-hot or soft)
            t: [B] or [B, 1] timestep values
            node_mask: [B, N] boolean mask for valid nodes
        
        Returns:
            [B, moment_dim] predicted scattering moments
        """
        B, N, _ = X_t.shape
        
        # Flatten edges and concatenate with nodes
        E_flat = E_t.reshape(B, N, -1)  # [B, N, N*Edim]
        x = torch.cat([X_t, E_flat], dim=-1)  # [B, N, Xdim + N*Edim]
        
        # Project to hidden size
        x = self.input_proj(x)  # [B, N, hidden_size]
        
        # Get timestep embedding
        t_emb = self.t_embedder(t)  # [B, hidden_size]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, node_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Global pooling: masked mean over nodes
        mask = node_mask.unsqueeze(-1).float()  # [B, N, 1]
        x_masked = x * mask
        pooled = x_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, hidden_size]
        
        # Predict moments
        moments = self.output_head(pooled)  # [B, moment_dim]
        
        return moments


def load_classifier(checkpoint_path: str, device: str = 'cuda') -> MomentClassifier:
    """
    Load a trained moment classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded MomentClassifier model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', checkpoint.get('model_config', {}))
    
    model = MomentClassifier(
        max_n_nodes=config['max_n_nodes'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        Xdim=config['Xdim'],
        Edim=config['Edim'],
        moment_dim=config['moment_dim'],
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def init_from_dit_checkpoint(
    classifier: MomentClassifier,
    dit_checkpoint_path: str,
    device: str = 'cuda',
    freeze_encoder: bool = False,
) -> MomentClassifier:
    """
    Initialize classifier encoder weights from a trained DiT checkpoint.
    
    Transfers learned graph processing weights from the DiT denoiser to give
    the classifier a head start on understanding noisy molecular graphs.
    
    Args:
        classifier: MomentClassifier to initialize
        dit_checkpoint_path: Path to DiT checkpoint
        device: Device for loading
        freeze_encoder: If True, freeze transferred weights
    
    Returns:
        Classifier with transferred weights
    """
    dit_checkpoint = torch.load(dit_checkpoint_path, map_location=device)
    dit_state = dit_checkpoint.get('model_state_dict', dit_checkpoint)
    
    # Map DiT weights to classifier weights
    # DiT structure: denoiser.blocks.{i}.attn, denoiser.blocks.{i}.mlp, etc.
    # Classifier structure: blocks.{i}.attn, blocks.{i}.mlp, etc.
    
    transferred = 0
    classifier_state = classifier.state_dict()
    
    for dit_key, dit_value in dit_state.items():
        # Try to map DiT encoder weights to classifier
        if 'denoiser.blocks' in dit_key:
            # Map: denoiser.blocks.X.attn -> blocks.X.attn (self-attention)
            # Skip cross-attention layers
            if 'cross_attn' in dit_key or 'norm_cross' in dit_key:
                continue
            
            # Remove 'denoiser.' prefix
            classifier_key = dit_key.replace('denoiser.', '')
            
            # Handle attention weight mapping
            # DiT uses AttentionWithNodeMask, classifier uses GraphSelfAttention
            # Both have similar structure but may have different weight names
            
            if classifier_key in classifier_state:
                if classifier_state[classifier_key].shape == dit_value.shape:
                    classifier_state[classifier_key] = dit_value
                    transferred += 1
                else:
                    print(f"Shape mismatch for {classifier_key}: "
                          f"{classifier_state[classifier_key].shape} vs {dit_value.shape}")
        
        # Also try to transfer timestep embedder weights
        elif 'denoiser.t_embedder' in dit_key:
            classifier_key = dit_key.replace('denoiser.', '')
            if classifier_key in classifier_state:
                if classifier_state[classifier_key].shape == dit_value.shape:
                    classifier_state[classifier_key] = dit_value
                    transferred += 1
    
    # Load the (potentially updated) state dict
    classifier.load_state_dict(classifier_state)
    
    print(f"Transferred {transferred} weight tensors from DiT checkpoint")
    
    # Optionally freeze encoder weights
    if freeze_encoder:
        for name, param in classifier.named_parameters():
            if 'blocks' in name or 't_embedder' in name or 'input_proj' in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        total = sum(p.numel() for p in classifier.parameters())
        print(f"Frozen encoder: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
    
    return classifier

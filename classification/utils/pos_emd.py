import torch
from matplotlib import pyplot as plt
import math
import torch.nn as nn


def decompose_matrix(A, d, niter=10):
    # Perform SVD
    U, S, V = torch.svd_lowrank(A, q=d, niter=niter)

    # Truncate U, S, and V to get dimensions N x d and d x N
    U_d = U[:, :d]
    S_d = S[:d]
    V_d = V[:, :d]

    # Construct the two matrices
    # Matrix 1 (N x d)
    M1 = U_d * torch.sqrt(S_d)
    # Matrix 2 (d x N)
    M2 = (V_d * torch.sqrt(S_d)).T
    return M1, M2

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, h, w):
        #x: [batch_size, h, w, dim]
        not_mask = torch.ones([x.shape[0], h, w],dtype=torch.bool,device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # [batch_size, h, w]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [64]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [64]

        pos_x = x_embed[:, :, :, None] / dim_t # [batch_size, h, w, 64]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)# [batch_size, h, w, 128]
        return pos

def visualize_attention_map(attn_map, figsize=None, save_path=None):
    """
    Visualizes a 2D attention map with a color bar and customizable figure size.

    Args:
    attn_map (torch.Tensor): A 2D tensor of shape (H, W).
    figsize (tuple): A tuple indicating the figure size.
    """
    if figsize is None:
        figsize = (attn_map.shape[1] // 100 + 1, attn_map.shape[0] // 100 + 1)
    if not isinstance(attn_map, torch.Tensor):
        raise ValueError("attn_map must be a torch.Tensor")
    if len(attn_map.shape) != 2:
        raise ValueError("attn_map must be 2D")

    # Convert to numpy for plotting
    attn_map_np = attn_map.numpy()

    # Create a figure and a set of subplots with specified figure size
    fig, ax = plt.subplots(figsize=figsize)

    # Display the attention map
    cax = ax.imshow(attn_map_np, cmap='viridis', aspect='auto')

    # Add a color bar
    fig.colorbar(cax, ax=ax)

    # Use tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()

class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """
    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


def f_p(x, p):
    """
    Compute f_p(x) = ||x|| / ||x^p|| * x^p

    Args:
        x: input tensor
        p: power parameter

    Returns:
        normalized tensor
    """
    # Compute x^p element-wise
    x_p = torch.pow(x, p)

    # Compute norms
    x_norm = torch.norm(x,dim=-1,keepdim=True)
    x_p_norm = torch.norm(x_p,dim=-1,keepdim=True)

    # Compute final result
    return (x_norm / x_p_norm) * x_p

def interpolate_position_embedding(x, h, w, dst_h, dst_w):
    x = x.view(-1, h, w, x.shape[1])
    x = x.permute(0, 3, 1, 2) # B, C, H, W
    x = nn.functional.interpolate(x, size=(dst_h, dst_w), mode='bicubic', align_corners=False)
    x = x.permute(0, 2, 3, 1)
    return x.flatten(0, 2)

def get_vis_example(h=128, w=128, temp=10000, basedim=64, scale=2*math.pi,
                    dst_h=56, dst_w=56):
    elu = nn.ELU()
    pos = PositionEmbeddingSine(basedim, temperature=temp, scale=scale)
    x = torch.ones(1, h, w, 128)
    pos_emd = pos(x, h, w)
    ex = pos_emd.flatten(0,2)
    # ex = elu(ex) + 1.0 # b, n, c
    z = 1 / (ex @ ex.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
    sim = ex@ex.T
    sim_ex = sim[(w//2)*h+h//2].softmax(dim=-1).view(h,w)
    #sim_ex = sim[-1].softmax(dim=-1).view(h,w)

    sim_ex2 = sim[(w//2)*h+h//2].view(h,w)
    sim_softmax = sim.softmax(dim=-1)
    #set the ele in sim_softmax to 0 if it is smaller than the mean value
    #sim_softmax = sim_softmax * (sim_softmax > 0.1*sim_softmax.mean())
    trancated_q, trancated_v = decompose_matrix(sim_softmax, 48)
    #trancated_q, trancated_v = ex, ex.T
    #trancated_q = f_p(trancated_q, 10)
    #trancated_v = f_p(trancated_v.transpose(-1,-2), 2).transpose(-1,-2)
    inter_q = interpolate_position_embedding(trancated_q, h, w, dst_h, dst_w)
    inter_v = interpolate_position_embedding(trancated_v.transpose(-1, -2), h, w, dst_h, dst_w).transpose(-1, -2)
    trancated_sim = trancated_q @ trancated_v
    trancated_ex = trancated_sim[(w // 2) * h + h // 2].view(h, w)
    #trancated_ex = trancated_sim[-1].view(h, w)

    inter_sim = inter_q @ inter_v
    inter_ex = inter_sim[(dst_w // 2) * dst_h + dst_h // 2].view(dst_h, dst_w)
    #inter_ex = inter_sim[-1].view(dst_h, dst_w)
    return sim_ex,inter_ex,trancated_ex


if __name__ == '__main__':
    sim_ex, sim_ex2,trancated_ex = get_vis_example(basedim=32, h=56, w=56, temp=10, scale=2*math.pi)

    #attention map from the center point after softmax
    visualize_attention_map(sim_ex, figsize=(6,6))

    #attention map from the center point before softmax
    #visualize_attention_map(sim_ex2, figsize=(6,6))

    #attention map from the center point after SVD
    visualize_attention_map(trancated_ex, figsize=(6,6))

    #attention map from the center point after interpolation
    visualize_attention_map(sim_ex2, figsize=(6,6))

    print('shape of sim_ex:', sim_ex.shape)
    pass

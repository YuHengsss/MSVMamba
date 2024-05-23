import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """
    Denormalize a tensor image with mean and standard deviation.
    :param tensor: image tensor to denormalize
    :param mean: mean used for normalization
    :param std: standard deviation used for normalization
    :return: denormalized image tensor
    """
    if tensor.ndim == 4:  # Batch of images [B, C, H, W]
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
    else:  # Single image [C, H, W]
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)

    # Denormalize
    tensor = tensor.cpu().detach().clone()
    tensor = (tensor * std + mean).clamp(0, 1)
    tensor = tensor.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    tensor = tensor.numpy()*255
    nparray = tensor.astype(int)
    return nparray


def visualize_batch(image_tensor,before_head=None):
    """
    Visualize a batch of images.
    :param image_tensor: image tensor to visualize
    :return: None
    """
    if image_tensor is not None:
        # Denormalize the image tensor
        image_tensor = denormalize(image_tensor) # [B, H, W, C]
        # Make a grid and show
        batch_size = image_tensor.shape[0]
        if batch_size > 1:
            # Create a grid of subplots
            fig, axs = plt.subplots(1, batch_size, figsize=(batch_size * 5, 5))
            for i in range(batch_size):
                axs[i].imshow(image_tensor[i])
                axs[i].axis('off')  # Hide axes ticks
            plt.title('Input Images')
            plt.show()
        else:
            plt.imshow(image_tensor[0])
            plt.title('Input Image')
            plt.show()

    if before_head is not None:
        #before_head with shape of [B,H,W,C]
        if type(before_head) is not list:
            before_heads = [before_head]
        else:
            before_heads = before_head
        fig,axs = plt.subplots(len(before_heads), 2 , figsize=(len(before_heads) * 5, 10))
        for i_layer,before_head in enumerate(before_heads):
            C = before_head.shape[-1]
            before_head = before_head.detach()
            l2_norm = torch.norm(before_head, p=2, dim=-1)[0]
            H, W = l2_norm.shape
            # for i in range(B):
            #     axs[i].imshow(l2_norm[i].cpu().numpy(), cmap='viridis')
            #     axs[i].axis('off')  # Hide axes ticks
            axs[i_layer,0].imshow(l2_norm.cpu().numpy(), cmap='viridis')
            axs[i_layer,0].axis('off')
            axs[i_layer,0].set_title('L2 Norm of feature maps in layer {}'.format(i_layer))


            first_feature_map = before_head[0].cpu().detach()
            max_norm_index = torch.argmax(l2_norm[0])
            max_feature = first_feature_map.view(-1,C)[max_norm_index]
            center_feature = first_feature_map[H//2, W//2]
            center_feature = center_feature.expand(H, W, C)
            cosine_sim = torch.nn.functional.cosine_similarity(first_feature_map, center_feature, dim=2)
            axs[i_layer,1].imshow(cosine_sim.numpy(), cmap='viridis')
            axs[i_layer,1].axis('off')
            axs[i_layer,1].set_title('Cosine Similarity with center feature in layer {}'.format(i_layer))
        # Adjust layout
        plt.tight_layout()
        plt.show()
    return
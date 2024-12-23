from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def vis_feature_map(A_mat, H, W, figure_size=(20, 20), colorbar_fontsize=20):
    '''
    A_mat: (B, heads, L, 1)
    '''

    B, nheads, L, _ = A_mat.shape
    # Move A_mat to CPU and convert to numpy
    A_mat = A_mat.cpu().detach().numpy()
    # Reshape A_mat to (B, nheads, H, W)
    A_mat = A_mat.reshape(B, nheads, H, W)

    # Calculate number of rows and columns for subplots
    n_cols = int(np.ceil(np.sqrt(nheads)))
    n_rows = int(np.ceil(nheads / n_cols))

    for i in range(B):
        plt.figure(figsize=figure_size)
        gs = GridSpec(n_rows, n_cols * 2, width_ratios=[1, 0.05] * n_cols)  # Alternate between plot and colorbar spaces
        #gs.update(wspace=0.5, hspace=0.5)  # Adjust spacing between subplots

        for i in range(B):
            plt.figure(figsize=figure_size)
            # Create a GridSpec for 2 columns per head: one for the image, one for the colorbar
            gs = GridSpec(n_rows, n_cols * 2)  # No need for width_ratios in this context

            for i in range(B):
                plt.figure(figsize=figure_size)
                # Adjust width_ratios to give less space to the colorbar
                gs = GridSpec(n_rows, n_cols * 2,
                              width_ratios=[10, 1] * n_cols)  # More space for images, less for colorbars

                for j in range(nheads):
                    row_index = j // n_cols
                    col_index = (j % n_cols) * 2  # Multiply by 2 to allocate space for the colorbar next to each plot

                    ax = plt.subplot(gs[row_index, col_index])  # Plot position
                    im = ax.imshow(A_mat[i, j], interpolation='nearest', cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f'Head {j + 1}', fontsize=colorbar_fontsize)

                    # # Create a color bar for each subplot
                    # cbar_ax = plt.subplot(gs[row_index, col_index + 1])  # Colorbar position
                    # plt.colorbar(im, cax=cbar_ax)
                    # cbar_ax.tick_params(labelsize=colorbar_fontsize)

                # Adjust subplot parameters to reduce spacing
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
                plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)
                plt.show()

            plt.tight_layout()
            plt.show()
    # show the average of all heads
    plt.figure(figsize=figure_size)
    plt.imshow(A_mat.mean(1).mean(0), interpolation='nearest', cmap='viridis')
    plt.axis('off')
    # plt.title('Average of all heads', fontsize=colorbar_fontsize)
    plt.tight_layout()
    plt.show()

    return




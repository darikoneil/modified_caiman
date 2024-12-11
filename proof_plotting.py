import matplotlib.pyplot as plt
import numpy as np


def raw_proof(raw_projection: np.ndarray,
              raw_correlation: np.ndarray,
              cmap: str = "RdYlBu_r",
              interp: str = "lanczos") -> None:
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    ax[0].imshow(raw_projection,
                 cmap=cmap,
                 vmin=np.percentile(np.ravel(raw_projection), 50),
                 vmax=np.percentile(np.ravel(raw_projection), 99.5),
                 interpolation=interp)
    ax[0].set_title("Max Projection (Raw)", fontsize=12)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].imshow(raw_correlation,
                 cmap=cmap,
                 vmin=np.percentile(np.ravel(raw_correlation), 50),
                 vmax=np.percentile(np.ravel(raw_correlation), 99.5),
                 interpolation=interp)
    ax[1].set_title('Correlation Image (Raw)', fontsize=12)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    plt.show()


def comparison_proof(projection_one: np.ndarray,
                     correlation_one: np.ndarray,
                     projection_two: np.ndarray,
                     correlation_two: np.ndarray,
                     title_one: str = "Stable",
                     title_two: str = "Raw",
                     cmap: str = "RdYlBu_r",
                     interp: str = "lanczos") -> None:

    fig, ax = plt.subplots(2,2, figsize=(6,6))
    ax[0,0].imshow(projection_one,
                   cmap=cmap,
                   vmin=np.percentile(np.ravel(projection_one), 50),
                   vmax=np.percentile(np.ravel(projection_one), 99.5),
                   interpolation=interp)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_title(f"Max Projection ({title_one.capitalize()})", fontsize=12)
    ax[0,1].imshow(correlation_one,
                   cmap=cmap,
                   vmin=np.percentile(np.ravel(correlation_one), 50),
                   vmax=np.percentile(np.ravel(correlation_one), 99.5),
                   interpolation=interp)
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].set_title(f"Correlation Image ({title_one.capitalize()}", fontsize=12)

    ax[1,0].imshow(projection_two,
                   cmap=cmap,
                   vmin=np.percentile(np.ravel(projection_two), 50),
                   vmax=np.percentile(np.ravel(projection_two), 99.5),
                   interpolation=interp)
    ax[1,0].set_title(f"Max Projection ({title_two.capitalize()}", fontsize=12)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,1].imshow(correlation_two,
                   cmap=cmap,
                   vmin=np.percentile(np.ravel(correlation_two), 50),
                   vmax=np.percentile(np.ravel(correlation_two), 99.5),
                   interpolation=interp)
    ax[1,1].set_title(f"Correlation ({title_two.capitalize()}", fontsize=12)
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])

    fig.tight_layout()
    plt.show()

"""Figure helpers: save a PNG and embed it in Jupyter."""

import os


def save_and_show(fig, name: str, figures_dir: str) -> str:
    """Save a figure to figures_dir and embed the PNG in the notebook output.

    Display the file (image/png), not the matplotlib Figure object. display(fig)
    under Agg / nbclient only stores the text '<Figure size ... with N Axes>'.
    """
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        from IPython.display import Image as IPyImage
        from IPython.display import display

        display(IPyImage(filename=os.path.abspath(path)))
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path

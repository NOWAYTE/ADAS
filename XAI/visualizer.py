import matplotlib.pyplot as plt

class visualizer:
    @staticmethod
    def plot_explanation(explanation, path=None, show=True):
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ax[0].imshow(explanation['explained'])
        ax[0].set_title(f"Class: {explanation['class_idx']} (Score: {explanation['score']:.2f})")
        ax[1].imshow(explanation['cam'])
        ax[1].set_title("Grad-CAM")

        ax[2].imshow(explanation['heatmap'], cmap='viridis')
        ax[2].set_title("Heatmap")
        ax[2].axis('off')
        if path:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
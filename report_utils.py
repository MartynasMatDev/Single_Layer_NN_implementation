import numpy as np
import matplotlib.pyplot as plt
from test_data import XTestSet
from training_data import vectors

def save_as_png(vectors, dir_prefix, width, output_dir="output_images"):
    import os
    os.makedirs(output_dir+dir_prefix, exist_ok=True)

    for idx, array in enumerate(vectors):
        grid = np.array(array).reshape((width, width))

        plt.imshow(grid, cmap="gray")
        plt.axis("off")

        # Save the image
        filename = os.path.join(output_dir + dir_prefix, f"test_x_{idx}.png")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    save_as_png(vectors, "/training" ,width=10)
    save_as_png(XTestSet, "/test" ,width=10)
    print("All images have been saved to the 'output_images' directory.")

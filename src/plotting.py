import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def show_single(val_dataset, preds, i):
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sources = np.concatenate([x for x, y in val_dataset], axis=0)
    targets = np.concatenate([y for x, y in val_dataset], axis=0)

    ax[0].imshow(sources[i], cmap="gray", vmin=-1, vmax=1)
    ax[0].set_title("Source")
    ax[1].imshow(targets[i], cmap="gray", vmin=-1, vmax=1)
    ax[1].set_title("Target")

    result = preds[i]
    ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
    ax[2].set_title("Result")

    difference = targets[i] - result
    ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
    ax[3].set_title("Difference")

    plt.suptitle(f"Example: {i:03d}")

    plt.show()


def plt_examples(val_dataset, preds):

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sources = np.concatenate([x for x, y in val_dataset], axis=0)
    targets = np.concatenate([y for x, y in val_dataset], axis=0)

    sc = ax[0].imshow(sources[0], cmap="gray", vmin=-1, vmax=1)
    ax[0].set_title("Source")
    tg = ax[1].imshow(targets[0], cmap="gray", vmin=-1, vmax=1)
    ax[1].set_title("Target")

    result = preds[0]
    re = ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
    ax[2].set_title("Result")

    difference = targets[0] - result
    dif = ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
    ax[3].set_title("Difference")

    suptitle = plt.suptitle(f"Example: ")

    def update(i):

        result = preds[i]
        difference = targets[i] - result
        
        sc.set_data(sources[i])
        tg.set_data(targets[i])
        re.set_data(result)
        dif.set_data(difference)

        suptitle.set_text(f"Example: {i:03d}")
        
        return sc, tg, re, dif, suptitle

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(preds))

    ani.save('many_examples.gif', writer='pillow', fps=1)


def plt_training(files, datasets, preds, i):

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sources = np.concatenate([x for x, y in datasets[0]], axis=0)
    targets = np.concatenate([y for x, y in datasets[0]], axis=0)

    sc = ax[0].imshow(sources[i], cmap="gray", vmin=-1, vmax=1)
    ax[0].set_title("Source")
    tg = ax[1].imshow(targets[i], cmap="gray", vmin=-1, vmax=1)
    ax[1].set_title("Target")

    result = preds[0][i]
    re = ax[2].imshow(result, cmap="gray", vmin=-1, vmax=1)
    ax[2].set_title("Result")

    difference = targets[i] - result
    dif = ax[3].imshow(difference, cmap="gray", vmin=-1, vmax=1)
    ax[3].set_title("Difference")

    suptitle = plt.suptitle(f"Epoch: {0:03d}")

    def update(j):

        sources = np.concatenate([x for x, y in datasets[j]], axis=0)
        targets = np.concatenate([y for x, y in datasets[j]], axis=0)
        result = preds[j][i]
        difference = targets[i] - result
        
        sc.set_data(sources[i])
        tg.set_data(targets[i])
        re.set_data(result)
        dif.set_data(difference)

        suptitle.set_text(files[j])
        
        return sc, tg, re, dif, suptitle

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(files))

    ani.save('train_animation.gif', writer='pillow', fps=3)


from PIL import Image
def export_image(h, filename, seismic=False):

    h = np.squeeze(h)
    normalized_data = (255 * (h - h.min()) / (h.max() - h.min())).astype(np.uint8)

    if not seismic:

        image = Image.fromarray(normalized_data.T, mode='L')
        image.save(filename)

    else:

        # Apply the seismic colormap
        colormap = plt.get_cmap('seismic')
        colored_image = colormap(normalized_data)

        # Convert the colormap output (which is RGBA) to RGB by multiplying by 255
        rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

        # Create an image from the RGB array
        image = Image.fromarray(rgb_image)
        image.save(filename)


# Updated function to create a GIF alternating between two images
def export_gif(image1, image2, filename, seismic=False, duration=500):
    
    # Normalize both images to 8-bit grayscale
    def normalize_image(h):
        h = np.squeeze(h)
        return (255 * (h - h.min()) / (h.max() - h.min())).astype(np.uint8)
    
    img1_normalized = normalize_image(image1)
    img2_normalized = normalize_image(image2)
    
    if not seismic:
        img1 = Image.fromarray(img1_normalized.T, mode='L')
        img2 = Image.fromarray(img2_normalized.T, mode='L')
    else:
        # Apply seismic colormap
        colormap = plt.get_cmap('seismic')
        img1_colored = colormap(img1_normalized)
        img2_colored = colormap(img2_normalized)

        img1 = Image.fromarray((img1_colored[:, :, :3] * 255).astype(np.uint8))
        img2 = Image.fromarray((img2_colored[:, :, :3] * 255).astype(np.uint8))

    # Create GIF with both images alternating
    img1.save(
        filename, save_all=True, append_images=[img2],
        duration=duration, loop=0
    )


def export_trace_img(trace1, trace2, filename, duration=500):

    samples = len(trace1)
    if len(trace2) != samples:
        raise ValueError(f"Trace 1 and Trace 2 must be the same length\
                         . Found lengths {samples} and {len(trace2)}")

    fig, ax = plt.subplots(1, 1, figsize=(15, 4.5))
    ax.plot(range(samples), trace1, color="red", linewidth=0.75, label="High Carrier")
    ax.plot(range(samples), trace2, color="blue", linewidth=0.75, label="Low Carrier")
    plt.savefig(filename)
    plt.close()
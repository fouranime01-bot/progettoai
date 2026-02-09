import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

def main():
    print("=== EDA MNIST ===")

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    print(f"Numero totale di immagini: {len(train_data)}")

    sample_img, sample_label = train_data[0]
    print(f"Shape immagine: {sample_img.shape}")
    print(f"Valore minimo: {sample_img.min().item():.4f}")
    print(f"Valore massimo: {sample_img.max().item():.4f}")

    labels = [label for _, label in train_data]
    unique, counts = np.unique(labels, return_counts=True)

    print("\nDistribuzione classi:")
    for u, c in zip(unique, counts):
        print(f"Classe {u}: {c} campioni")

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img, label = train_data[i]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

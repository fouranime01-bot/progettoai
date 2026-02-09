import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms


CLASS_NAMES = {
    0: "T-shirt/Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}


def main():

    print("=== EDA: Fashion-MNIST Dataset ===")

    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
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
        print(f"Classe {u} ({CLASS_NAMES[u]}): {c} campioni")

    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color="steelblue")
    plt.title("Distribuzione delle classi (Fashion-MNIST)")
    plt.xlabel("Classe")
    plt.ylabel("Numero di campioni")
    plt.xticks(
        unique,
        [CLASS_NAMES[u] for u in unique],
        rotation=45,
        ha="right"
    )
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img, label = train_data[i]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"{CLASS_NAMES[label]}")
        ax.axis("off")

    plt.suptitle("Esempi dal dataset Fashion-MNIST", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

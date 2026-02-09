import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.model import SimpleNet


def train():
    print("Inizio training...")

    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = SimpleNet()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        for batch_idx, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "artifacts/model.pt")
    print("Modello salvato in artifacts/model.pt")


if __name__ == "__main__":
    train()

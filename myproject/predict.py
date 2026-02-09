import torch
from torchvision import datasets, transforms
from myproject.model import SimpleNet

def predict():
    print("Caricamento modello...")
    model = SimpleNet()
    model.load_state_dict(torch.load("artifacts/model.pt"))
    model.eval()

    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    x, y = test_data[0]  # prima immagine del test set
    x = x.unsqueeze(0)   # aggiunge batch dimension

    with torch.no_grad():
        pred = model(x)
        predicted_class = pred.argmax(dim=1).item()

    print(f"Predizione: {predicted_class}")

if __name__ == "__main__":
    predict()

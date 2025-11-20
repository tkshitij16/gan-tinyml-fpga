import torch, numpy as np, os
from torchvision import transforms
from PIL import Image

class TinyStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # tiny 3-layer conv net as a placeholder "generator"
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 1, 3, padding=1),
            torch.nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

def main():
    os.makedirs("logs/golden", exist_ok=True)
    model = TinyStub().eval()

    # fake edge image
    img = Image.new("L", (128, 128), color=128)
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0)  # [1,1,128,128]

    with torch.no_grad():
        y = model(x)

    np.savez("logs/golden/golden_io.npz", x=x.numpy(), y=y.numpy())
    out = transforms.ToPILImage()(y.squeeze(0))
    out.save("logs/golden/out_stub.png")
    print("Saved golden I/O to logs/golden")

if __name__ == "__main__":
    main()

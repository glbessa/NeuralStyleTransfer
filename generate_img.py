import torch
from torch import optim
from torchvision.utils import save_image

from train import VGG, load_image

if __name__ == '__main__':
    inputfile = 'gabriel.jpeg'
    outputfile = 'generated.png'

    checkpoint = torch.load("checkpoint.pth")
    model = VGG()
    optimizer = optim.Adam(model.parameters(), lr=0)

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])

    image = load_image(inputfile)
    generated = None

    with torch.no_grad():
        generated = model(image)

    save_image(generated, outputfile)
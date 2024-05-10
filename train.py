# Loss function
#   Ltotal(G) = a * Lcontent(C, G) + b * Lstyle(S, G)
#   Lcontent(C, G) = 1/2 * ||a[l](C) - a[l](G)||
#   Lstyle(S, G) -> Gram Matrix for Style and Generated Image

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {'state_dict': None }

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

if __name__ == '__main__':
    checkpoint = {
        'model_state':None,
        'optim_state':None
    }

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    image_size = 356

    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    original_img = load_image('gabriel.jpeg')
    style_img = load_image('noite-estrelada-van-gogh.jpeg')

    model = VGG().to(device).eval()
    generated = original_img.clone().requires_grad_(True)

    # Hyperparameters
    total_steps = 6000
    learning_rate = 1e-3
    alpha = 1
    beta = 1e-2
    optimizer = optim.Adam([generated], lr=learning_rate)

    for step in range(total_steps):
        generated_features = model(generated)
        original_img_features = model(original_img)
        style_features = model(style_img)

        style_loss = original_loss = 0

        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_img_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape
            original_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # Compute Gram Matrix
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )

            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )

            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(total_loss)
            save_image(generated, "generated.png")
            checkpoint['model_state'] = model.state_dict()
            checkpoint['optim_state'] = optimizer.state_dict()
            torch.save(checkpoint, "checkpoint.pth")

    save_image(generated, "generated.png")
    checkpoint['model_state'] = model.state_dict()
    checkpoint['optim_state'] = optimizer.state_dict()
    torch.save(checkpoint, "checkpoint.pth")

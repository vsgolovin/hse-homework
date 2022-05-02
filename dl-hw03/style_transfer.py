import os
from PIL import Image
from torchvision import transforms, models
import torch
from torch import nn


CONT_IMG_PATH = os.path.join(os.getcwd(), 'data', 'content')
STYLE_IMG_PATH = os.path.join(os.getcwd(), 'data', 'style')
OUTPUT_PATH = os.path.join(os.getcwd(), 'results')
IMG_SIZE = 512
loader = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


def main():
    content_imgs = [f for f in os.scandir(CONT_IMG_PATH) if f.is_file()]
    style_imgs = [f for f in os.scandir(STYLE_IMG_PATH) if f.is_file()]

    for content_img in content_imgs:
        name_c = content_img.name.split('.')[0]
        img_c = load_img(content_img.path)

        for style_img in style_imgs:
            name_s = style_img.name.split('.')[0]
            img_s = load_img(style_img.path)

            outfile = os.path.join(OUTPUT_PATH, f'{name_c}-{name_s}.jpg')
            print(outfile)
            if os.path.exists(outfile):
                continue

            output = style_transfer(
                content_image=img_c,
                style_image=img_s,
                content_layer_inds=[13],
                content_loss_weights=[1.0],
                style_layer_inds=[0, 2, 4, 8, 12],
                style_loss_weights=[2e4] * 5,
                num_iter=1000,
            )
            output.save(outfile)


def load_img(fname):
    img = Image.open(fname)
    return loader(img).to(DEVICE)


def tensor2image(x):
    return unloader(x.detach().cpu())


def calculate_content_loss(x, target):
    return nn.functional.mse_loss(x, target.detach())


def gram_matrix(x):
    a = 1
    b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def calculate_style_loss(x, target):
    Gx = gram_matrix(x)
    Gt = gram_matrix(target.detach())
    return nn.functional.mse_loss(Gx, Gt)


class StyleContentModel(nn.Module):
    def __init__(self, content_indices, content_weights,
                 style_indices, style_weights):
        super().__init__()
        # get pretrained VGG model
        vgg = models.vgg19(pretrained=True).features.to(DEVICE)
        self.vgg = nn.Sequential()

        # find absolute indices of layers needed for calculating content
        # and style losses
        self.content_inds = []
        self.style_inds = []
        conv_count = 0
        for name, layer in vgg.named_children():
            ci = None
            si = None
            if isinstance(layer, nn.Conv2d):
                if conv_count in content_indices:
                    ci = content_indices.index(conv_count)
                if conv_count in style_indices:
                    si = style_indices.index(conv_count)
                conv_count += 1
            elif isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            self.vgg.add_module(name, layer)
            self.content_inds.append(ci)
            self.style_inds.append(si)
        # lists will look like [None, None, 0, None, 1, ...]

        # do not update vgg weights
        self.vgg.eval()
        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)

        # store weights for calculating loss
        self.content_weights = content_weights
        self.style_weights = style_weights

        # VGG requires input normalization
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

    def forward(self, x, tc, ts):
        # normalize inputs
        x = self.normalizer(x)
        tc = self.normalizer(tc)
        ts = self.normalizer(ts)

        content_losses = []
        style_losses = []

        # go through VGG layers
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            tc = layer(tc)
            ts = layer(ts)

            # update content and style losses if needed
            if self.content_inds[i] is not None:
                w = self.content_weights[self.content_inds[i]]
                content_losses.append(
                    w * calculate_content_loss(x, tc))
            if self.style_inds[i] is not None:
                w = self.style_weights[self.style_inds[i]]
                style_losses.append(w * calculate_style_loss(x, ts))

        return (torch.sum(torch.stack(content_losses)),
                torch.sum(torch.stack(style_losses)))


def style_transfer(content_image, style_image,
                   content_layer_inds, style_layer_inds,
                   content_loss_weights, style_loss_weights,
                   num_iter=100, **kwargs):
    def closure():
        with torch.no_grad():
            input_image.clamp_(0, 1)
        optimizer.zero_grad()
        content_loss, style_loss = model.forward(input_image,
                                                 content_image, style_image)
        loss = content_loss + style_loss
        loss.backward()
        iter_count[0] += 1

        s = '\rIteration: {}, content_loss: {:.2e}, style_loss: {:.2e}'.format(
            iter_count[0], content_loss.item(), style_loss.item()
        )
        print(s, end='')
        return loss

    input_image = content_image.clone()
    # input_image = torch.rand_like(content_image)
    input_image.requires_grad_(True)
    model = StyleContentModel(content_layer_inds, content_loss_weights,
                              style_layer_inds, style_loss_weights)
    optimizer = torch.optim.LBFGS(params=[input_image], **kwargs)
    iter_count = [0]
    while iter_count[0] < num_iter:
        optimizer.zero_grad()
        optimizer.step(closure=closure)
    print()

    with torch.no_grad():
        input_image.clamp_(0, 1)
    return tensor2image(input_image)


if __name__ == '__main__':
    main()

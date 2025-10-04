import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(torch.flatten(x, start_dim=1))

    def getPLR(self, x):
        return self.forward(x)


class DNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(256, 128), act=torch.nn.Tanh, dropout=0.1):
        super().__init__()
        dims = (input_dim, *hidden)
        layers = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(i, o), act()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
        self.feat = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        self.head = torch.nn.Linear(hidden[-1] if hidden else input_dim, output_dim)

    def forward(self, x):
        x = x.flatten(1)
        return self.head(self.feat(x))

    def getPLR(self, x):
        x = x.flatten(1)
        return self.feat(x)


if __name__ == "__main__":
    net = DNN(input_dim=10, output_dim=10)
    net.eval()
    torch.manual_seed(1)
    x   = torch.randn(10, 10)
    out = net(x)
    print("len(out):", len(out))
    print("out.shape:", out.shape)

    import torch
    from models import resnet

    print('torch version', torch.__version__)

    # CIFAR10
    shape = resnet.infer_image_shape(3*32*32, "CIFAR10")
    print('CIFAR inferred shape', shape)
    net = resnet.ResNetClassifier('resnet20', shape, num_classes=10, pretrained=False)
    x = torch.zeros(2, 3*32*32)
    out = net(x)
    print('Output shape CIFAR resnet20:', out.shape)

    # CIFAR100 resnet20
    shape100 = resnet.infer_image_shape(3*32*32, "CIFAR100")
    net100 = resnet.ResNetClassifier('resnet20', shape100, num_classes=100, pretrained=False)
    x100 = torch.zeros(2, 3*32*32)
    out100 = net100(x100)
    print('Output shape CIFAR100 resnet20:', out100.shape)

    # MNIST
    shape_mnist = resnet.infer_image_shape(28*28, 'MNIST')
    net_mnist = resnet.ResNetClassifier('resnet18', shape_mnist, num_classes=10, pretrained=False)
    xm = torch.zeros(2, 28*28)
    outm = net_mnist(xm)
    print('Output shape MNIST resnet18:', outm.shape)

    # FEMNIST
    shape_femnist = resnet.infer_image_shape(28*28, 'FEMNIST')
    net_fem = resnet.ResNetClassifier('resnet18', shape_femnist, num_classes=62, pretrained=False)
    xf = torch.zeros(2, 28*28)
    outf = net_fem(xf)
    print('Output shape FEMNIST resnet18:', outf.shape)
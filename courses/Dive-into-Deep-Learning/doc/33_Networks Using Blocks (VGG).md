## Networks Using Blocks (VGG)

While AlexNet proved that deep convolutional neural networks
can achieve good results, it did not offer a general template
to guide subsequent researchers in designing new networks.
In the following sections, we will introduce several heuristic concepts
commonly used to design deep networks.

Progress in this field mirrors that in chip design
where engineers went from placing transistors
to logical elements to logic blocks.
Similarly, the design of neural network architectures
had grown progressively more abstract,
with researchers moving from thinking in terms of
individual neurons to whole layers,
and now to blocks, repeating patterns of layers.

The idea of using blocks first emerged from the
[Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG)
at Oxford University,
in their eponymously-named VGG network.
It is easy to implement these repeated structures in code
with any modern deep learning framework by using loops and subroutines.

### VGG Blocks

The basic building block of classic convolutional networks
is a sequence of the following layers:
(i) a convolutional layer
(with padding to maintain the resolution),
(ii) a nonlinearity such as a ReLU, (iii) a pooling layer such
as a max pooling layer.
One VGG block consists of a sequence of convolutional layers,
followed by a max pooling layer for spatial downsampling.
In the original VGG paper (`Simonyan.Zisserman.2014`),
the authors
employed convolutions with $3\times3$ kernels
and $2 \times 2$ max pooling with stride of $2$
(halving the resolution after each block).
In the code below, we define a function called `vgg_block`
to implement one VGG block.
The function takes two arguments
corresponding to the number of convolutional layers `num_convs`
and the number of output channels `num_channels`.

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms



def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)



vgg_block(3,3,10)

### VGG Network

Like AlexNet and LeNet,
the VGG Network can be partitioned into two parts:
the first consisting mostly of convolutional and pooling layers
and a second consisting of fully-connected layers.
The convolutional portion of the net connects several `vgg_block` modules
in succession.
In the following figure, the variable `conv_arch` consists of a list of tuples (one per block),
where each contains two values: the number of convolutional layers
and the number of output channels,
which are precisely the arguments requires to call
the `vgg_block` function.
The fully-connected module is identical to that covered in AlexNet.

![Designing a network from building blocks](https://doc.shiyanlou.com/courses/2777/246442/a4011bf15cb3c5cddad5674de814171b-1)

The original VGG network had 5 convolutional blocks,
among which the first two have one convolutional layer each
and the latter three contain two convolutional layers each.
The first block has 64 output channels
and each subsequent block doubles the number of output channels,
until that number reaches $512$.
Since this network uses $8$ convolutional layers
and $3$ fully-connected layers, it is often called VGG-11.

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))



The following code implements VGG-11. 
This is a simple matter of executing a for loop over `conv_arch`.

def vgg(conv_arch):
    # The convulational layer part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels*7*7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)



To make the model easy to train on `cpu`, we change the size of the image, from $224*224$ to $64*64$, therefore, the input size of linear need to be changed too. 

def vgg(conv_arch):
    # The convulational layer part
    conv_blks=[]
    in_channels=1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels*2*2, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(1024, 10))

net = vgg(conv_arch)



Next, we will construct a single-channel data example
with a height and width of 224 to observe the output shape of each layer.


X = torch.randn(size=(1, 1, 64, 64))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)



As you can see, we halve height and width at each block,
finally reaching a height and width of 7
before flattening the representations
for processing by the fully-connected layer.

### Function Preparation

We will define several functions here. 
Because all these functions we have used before, so we can use them directly.

# define the functions for visualization
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

import time

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator: 
    """For accumulating sums over `n` variables.
    """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device=try_gpu()):
    """Train and evaluate a model with CPU or GPU."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0.01, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i/len(train_iter), (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

### Model Training

Since VGG-11 is more computationally-heavy than AlexNet
we construct a network with a smaller number of channels.
This is more than sufficient for training on Fashion-MNIST.

ratio = 16
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)



We prepare the dataset for the training.

!wget -nc "https://labfile.oss.aliyuncs.com/courses/2777/FashionMNIST.zip"
!unzip -o "FashionMNIST.zip"

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=1),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=1))

Apart from using a slightly larger learning rate,
the model training process is similar to that of AlexNet in the last section.
The training will take about 10-20 minutes.

lr, num_epochs, batch_size = 0.05, 10, 128,
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=64)
train_ch6(net, train_iter, test_iter, num_epochs, lr)



### Summary

* VGG-11 constructs a network using reusable convolutional blocks. Different VGG models can be defined by the differences in the number of convolutional layers and output channels in each block.
* The use of blocks leads to very compact representations of the network definition. It allows for efficient design of complex networks.
* In their work Simonyan and Ziserman experimented with various architectures. In particular, they found that several layers of deep and narrow convolutions (i.e., $3 \times 3$) were more effective than fewer layers of wider convolutions.

### Exercises

1. When printing out the dimensions of the layers we only saw 8 results rather than 11. Where did the remaining 3 layer informations go?
1. Compared with AlexNet, VGG is much slower in terms of computation, and it also needs more GPU memory. Try to analyze the reasons for this.
1. Try to change the height and width of the images in Fashion-MNIST from 224 to 96. What influence does this have on the experiments?
1. Refer to Table 1 in :cite:`Simonyan.Zisserman.2014` to construct other common models, such as VGG-16 or VGG-19.

[Discussions](https://discuss.d2l.ai/t/78)

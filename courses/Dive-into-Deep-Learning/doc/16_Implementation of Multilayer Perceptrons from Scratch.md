## Implementation of Multilayer Perceptrons from Scratch

Now that we have characterized 
multilayer perceptrons (MLPs) mathematically, 
let us try to implement one ourselves.

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display
from torch.utils import data
from torchvision import transforms
import torchvision
import torch
from torch import nn



To compare against our previous results
achieved with softmax regression,
we will continue to work with 
the Fashion-MNIST image classification dataset. 
We download the dataset first.

!wget -nc "https://labfile.oss.aliyuncs.com/courses/2777/FashionMNIST.zip"
!unzip -o "FashionMNIST.zip"



Then we load the data into memory. 
We define the load function first.
We can use this function directly.

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

After that, we use `load_data_fashion_mnist` ro load the fashion mnist dataset.

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)



### Initializing Model Parameters

Recall that Fashion-MNIST contains 10 classes,
and that each image consists of a $28 \times 28 = 784$
grid of grayscale pixel values.
Again, we will disregard the spatial structure
among the pixels for now,
so we can think of this as simply a classification dataset
with 784 input features and 10 classes.

To begin, we will implement an MLP
with one hidden layer and 256 hidden units.
Note that we can regard both of these quantities
as hyperparameters.
Typically, we choose layer widths in powers of 2,
which tend to be computationally efficient because
of how memory is allocated and addressed in hardware.

Again, we will represent our parameters with several tensors.
Note that *for every layer*, we must keep track of
one weight matrix and one bias vector.
As always, we allocate memory
for the gradients of the loss with respect to these parameters.

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
print(W1.shape, b1.shape, W2.shape, b2.shape)



### Activation Function

To make sure we know how everything works,
we will implement the ReLU activation ourselves
using the maximum function rather than 
invoking the built-in `relu` function directly.

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)



We can test this `relu` function.

x = torch.tensor([-1,-2,1,2,3,4])
relu(x)



### Model

Because we are disregarding spatial structure, 
we `reshape` each two-dimensional image into 
a flat vector of length  `num_inputs`.
Finally, we implement our model 
with just a few lines of code.

def net(X):
    X = torch.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)



### Loss Function

To ensure numerical stability,
and because we already implemented
the softmax function from scratch,
we leverage the integrated function from high-level APIs
for calculating the softmax and cross-entropy loss.
But we encourage the interested reader 
to examine the source code for the loss function
to deepen their knowledge of implementation details.

loss = nn.CrossEntropyLoss()



### Training

Fortunately, the training loop for MLPs
is exactly the same as for softmax regression.
Therefore, we can use the functions at the section. 
Here we define these functions first. 
We define the functions for visualization first.

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

Then we define the functions for evaluate accuracy for a mode.

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

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset.
    """
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if not device:
        device = next(iter(net.parameters())).device
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]

Finally, we define the functions to train our model.

def train_epoch_ch3(net, train_iter, loss, updater):
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.size().numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

Now, we can train our model use the predefined functions.
Here we need to wait for about 5-10 minutes.

num_epochs, lr = 10, 0.5
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)



To evaluate the learned model, 
we apply it on some test data.
We define some functions first. 
Similarly, we can use them directly.

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def predict_ch3(net, test_iter, n=6):
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)



### Summary

* We saw that implementing a simple MLP is easy, even when done manually.
* However, with a large number of layers, implementing MLPs from scratch can still get messy (e.g., naming and keeping track of our model's parameters).

### Exercises

1. Change the value of the hyperparameter `num_hiddens` and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others constant.
1. Try adding an additional hidden layer to see how it affects the results.
1. How does changing the learning rate alter your results? Fixing the model architecture and other hyperparameters (including number of epochs), what learning rate gives you the best results? 
1. What is the best result you can get by optimizing over all the hyperparameters (learning rate, number of epochs, number of hidden layers, number of hidden units per layer) jointly? 
1. Describe why it is much more challenging to deal with multiple hyperparameters. 
1. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?

[Discussions](https://discuss.d2l.ai/t/93)

## The Image Classification Dataset

One of the widely used dataset for image classification is the  MNIST dataset (`LeCun.Bottou.Bengio.ea.1998`).
While it had a good run as a benchmark dataset,
even simple models by today's standards achieve classification accuracy over 95%,
making it unsuitable for distinguishing between stronger models and weaker ones.
Today, MNIST serves as more of sanity checks than as a benchmark.
To up the ante just a bit, we will focus our discussion in the coming sections
on the qualitatively similar, but comparatively complex Fashion-MNIST
dataset (`Xiao.Rasul.Vollgraf.2017`), which was released in 2017.

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils import data



### Reading the Dataset

We can download and read the Fashion-MNIST dataset into memory via the the build-in functions in the framework. We will download this dataset from Aliyun first.

!wget -nc "https://labfile.oss.aliyuncs.com/courses/2777/FashionMNIST.zip"
!unzip -o "FashionMNIST.zip"



Then we read the Fashion-MNIST.

# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./", train=False, transform=trans, download=True)



Fashion-MNIST consists of images from 10 categories, each represented
by 6000 images in the training set and by 1000 in the test set.
Consequently the training set and the test set
contain 60000 and 10000 images, respectively.


len(mnist_train), len(mnist_test)



The height and width of each input image are both 28 pixels.
Note that the dataset consists of grayscale images, whose number of channels is 1.
For brevity, throughout this book
we store the shape of any image with height $h$ width $w$ pixels as $h \times w$ or ($h$, $w$).


mnist_train[0][0].shape



The images in Fashion-MNIST are associated with the following categories:
t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.
The following function converts between numeric label indices and their names in text.


def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]



We can now create a function to visualize these examples.


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



Here are the images and their corresponding labels (in text)
for the first few examples in the training dataset.


X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));



### Reading a Minibatch

To make our life easier when reading from the training and test sets,
we use the built-in data iterator rather than creating one from scratch.
Recall that at each iteration, a load loader
reads a minibatch of data with size `batch_size` each time.
We also randomly shuffle the examples for the training data iterator.

batch_size = 256

def get_dataloader_workers():
    """Use 1 processes to read the data."""
    return 1

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())



Let us look at the time it takes to read the training data.
We define the class to record time first. 
This `Timer` has been used before, therefore we can use it directly here.

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

timer = Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'



### Putting All Things Together

Now we define the `load_data_fashion_mnist` function
that obtains and reads the Fashion-MNIST dataset.
It returns the data iterators for both the training set and validation set.
In addition, it accepts an optional argument to resize images to another shape.

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # trans is a List, use insert to insert item in the list
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))



Below we test the image resizing feature of the `load_data_fashion_mnist` function
by specifying the `resize` argument.


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break



We are now ready to work with the Fashion-MNIST dataset in the sections that follow.

### Summary

* Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. We will use this dataset in subsequent sections and chapters to evaluate various classification algorithms.
* We store the shape of any image with height $h$ width $w$ pixels as $h \times w$ or ($h$, $w$).
* Data iterators are a key component for efficient performance. Rely on well-implemented data iterators that exploit high-performance computing to avoid slowing down your training loop.

### Exercises

1. Does reducing the `batch_size` (for instance, to 1) affect the reading performance?
1. The data iterator performance is important. Do you think the current implementation is fast enough? Explore various options to improve it.
1. Check out the framework's online API documentation. Which other datasets are available?

[Discussions](https://discuss.d2l.ai/t/49)

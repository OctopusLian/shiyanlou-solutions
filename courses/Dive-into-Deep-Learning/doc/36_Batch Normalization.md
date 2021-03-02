## Batch Normalization

Training deep neural nets is difficult.
And getting them to converge in a reasonable amount of time can be tricky.
In this section, we describe batch normalization (BN)
(`Ioffe.Szegedy.2015`), a popular and effective technique
that consistently accelerates the convergence of deep nets.
Together with residual blocks, which will be covered in the following section—BN
has made it possible for practitioners
to routinely train networks with over 100 layers.

### Training Deep Networks

To motivate batch normalization, let us review
a few practical challenges that arise
when training ML models and neural nets in particular.

1. Choices regarding data preprocessing often
   make an enormous difference in the final results.
   Recall our application of multilayer perceptrons
   to predicting house prices.
   Our first step when working with real data
   was to standardize our input features
   to each have a mean of *zero* and variance of *one*.
   Intuitively, this standardization plays nicely with our optimizers
   because it puts the  parameters are a-priori at a similar scale.
1. For a typical MLP or CNN, as we train,
   the activations in intermediate layers
   may take values with widely varying magnitudes—both
   along the layers from the input to the output,
   across nodes in the same layer,
   and over time due to our updates to the model's parameters.
   The inventors of batch normalization postulated informally
   that this drift in the distribution of activations
   could hamper the convergence of the network.
   Intuitively, we might conjecture that if one
   layer has activation values that are 100x that of another layer,
   this might necessitate compensatory adjustments in the learning rates.
1. Deeper networks are complex and easily capable of overfitting.
   This means that regularization becomes more critical.

Batch normalization is applied to individual layers
(optionally, to all of them) and works as follows:
In each training iteration,
we first normalize the inputs (of batch normalization)
by subtracting their mean and
dividing by their standard deviation,
where both are estimated based on the statistics of the current minibatch.
Next, we apply a scaling coefficient and a scaling offset.
It is precisely due to this *normalization* based on *batch* statistics
that *batch normalization* derives its name.

Note that if we tried to apply BN with minibatches of size $1$,
we would not be able to learn anything.
That is because after subtracting the means,
each hidden node would take value $0$!
As you might guess, since we are devoting a whole section to BN,
with large enough minibatches, the approach proves effective and stable.
One takeaway here is that when applying BN,
the choice of minibatch size may be
even more significant than without BN.

Formally, BN transforms the activations at a given layer $\mathbf{x}$
according to the following expression:

$$\mathrm{BN}(\mathbf{x}) = \mathbf{\gamma} \odot \frac{\mathbf{x} - \hat{\mathbf{\mu}}}{\hat\sigma} + \mathbf{\beta}$$

Here, $\hat{\mathbf{\mu}}$ is the minibatch sample mean
and $\hat{\mathbf{\sigma}}$ is the minibatch sample standard deviation.
After applying BN, the resulting minibatch of activations
has zero mean and unit variance.
Because the choice of unit variance
(vs some other magic number) is an arbitrary choice,
we commonly include coordinate-wise
scaling coefficients $\mathbf{\gamma}$ and offsets $\mathbf{\beta}$.
Consequently, the activation magnitudes
for intermediate layers cannot diverge during training
because BN actively centers and rescales them back
to a given mean and size (via $\mathbf{\mu}$ and $\sigma$).
One piece of practitioner's intuition/wisdom
is that BN seems to allows for more aggressive learning rates.

Formally, denoting a particular minibatch by $\mathcal{B}$,
we calculate $\hat{\mathbf{\mu}}_\mathcal{B}$ and $\hat\sigma_\mathcal{B}$ as follows:

$$\hat{\mathbf{\mu}}_\mathcal{B} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\text{ and }
\hat{\mathbf{\sigma}}_\mathcal{B}^2 \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \mathbf{\mu}_{\mathcal{B}})^2 + \epsilon$$

Note that we add a small constant $\epsilon > 0$
to the variance estimate
to ensure that we never attempt division by zero,
even in cases where the empirical variance estimate might vanish.
The estimates $\hat{\mathbf{\mu}}_\mathcal{B}$
and $\hat{\mathbf{\sigma}}_\mathcal{B}$ counteract the scaling issue
by using noisy estimates of mean and variance.
You might think that this noisiness should be a problem.
As it turns out, this is actually beneficial.

This turns out to be a recurring theme in deep learning.
For reasons that are not yet well-characterized theoretically,
various sources of noise in optimization
often lead to faster training and less overfitting.
While traditional machine learning theorists
might buckle at this characterization,
this variation appears to act as a form of regularization.
In some preliminary research,
like `Teye.Azizpour.Smith.2018` and `Luo.Wang.Shao.ea.2018`
relate the properties of BN to Bayesian Priors and penalties respectively.
In particular, this sheds some light on the puzzle
of why BN works best for moderate minibatches sizes in the $50$–$100$ range.

Fixing a trained model, you might (rightly) think
that we would prefer to use the entire dataset
to estimate the mean and variance.
Once training is complete, why would we want
the same image to be classified differently,
depending on the batch in which it happens to reside?
During training, such exact calculation is infeasible
because the activations for all data points
change every time we update our model.
However, once the model is trained,
we can calculate the means and variances
of each layer's activations based on the entire dataset.
Indeed this is standard practice for
models employing batch normalization
and thus BN layers function differently
in *training mode* (normalizing by minibatch statistics)
and in *prediction mode* (normalizing by dataset statistics).

We are now ready to take a look at how batch normalization works in practice.

### Batch Normalization Layers

Batch normalization implementations for fully-connected layers
and convolutional layers are slightly different.
We discuss both cases below.
Recall that one key differences between BN and other layers
is that because BN operates on a full minibatch at a time,
we cannot just ignore the batch dimension
as we did before when introducing other layers.

### Fully-Connected Layers

When applying BN to fully-connected layers,
we usually insert BN after the affine transformation
and before the nonlinear activation function.
Denoting the input to the layer by $\mathbf{x}$,
the linear transform (with weights $\theta$) by $f_{\theta}(\cdot)$,
the activation function by $\phi(\cdot)$,
and the BN operation with parameters $\mathbf{\beta}$ and $\mathbf{\gamma}$
by $\mathrm{BN}_{\mathbf{\beta}, \mathbf{\gamma}}$,
we can express the computation of a BN-enabled,
fully-connected layer $\mathbf{h}$ as follows:

$$\mathbf{h} = \phi(\mathrm{BN}_{\mathbf{\beta}, \mathbf{\gamma}}(f_{\mathbf{\theta}}(\mathbf{x}) ) ) $$

Recall that mean and variance are computed
on the *same* minibatch $\mathcal{B}$
on which the transformation is applied.
Also recall that the scaling coefficient $\mathbf{\gamma}$
and the offset $\mathbf{\beta}$ are parameters that need to be learned
jointly with the more familiar parameters $\mathbf{\theta}$.

### Convolutional Layers

Similarly, with convolutional layers,
we typically apply BN after the convolution
and before the nonlinear activation function.
When the convolution has multiple output channels,
we need to carry out batch normalization
for *each* of the outputs of these channels,
and each channel has its own scale and shift parameters,
both of which are scalars.
Assume that our minibatches contain $m$ each
and that for each channel,
the output of the convolution has height $p$ and width $q$.
For convolutional layers, we carry out each batch normalization
over the $m \cdot p \cdot q$ elements per output channel simultaneously.
Thus we collect the values over all spatial locations
when computing the mean and variance
and consequently (within a given channel)
apply the same $\hat{\mathbf{\mu}}$ and $\hat{\mathbf{\sigma}}$
to normalize the values at each spatial location.

### Batch Normalization During Prediction

As we mentioned earlier, BN typically behaves differently
in training mode and prediction mode.
First, the noise in $\mathbf{\mu}$ and $\mathbf{\sigma}$
arising from estimating each on minibatches
are no longer desirable once we have trained the model.
Second, we might not have the luxury
of computing per-batch normalization statistics, e.g.,
we might need to apply our model to make one prediction at a time.

Typically, after training, we use the entire dataset
to compute stable estimates of the activation statistics
and then fix them at prediction time.
Consequently, BN behaves differently during training and at test time.
Recall that dropout also exhibits this characteristic.

### Implementation from Scratch

Below, we implement a batch normalization layer with tensors from scratch:

import torch
from torch import nn



def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use torch.is_grad_enabled() to determine whether the current mode is
    # training mode or prediction mode
    if not torch.is_grad_enabled(): # 预测模式就不需要计算均值和方差
        # If it is the prediction mode, directly use the mean and variance
        # obtained from the incoming moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4) # 长度=2, 全连接; 长度=4, 卷积;
        if len(X.shape) == 2: 
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcast
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance of the moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var



We can now create a proper `BatchNorm` layer.
Our layer will maintain proper parameters
corresponding for scale `gamma` and shift `beta`,
both of which will be updated in the course of training.
Additionally, our layer will maintain
a moving average of the means and variances
for subsequent use during model prediction.

Putting aside the algorithmic details,
note the design pattern underlying our implementation of the layer.
Typically, we define the math in a separate function, say `batch_norm`.
We then integrate this functionality into a custom layer,
whose code mostly addresses bookkeeping matters,
such as moving data to the right device context,
allocating and initializing any required variables,
keeping track of running averages (here for mean and variance), etc.
This pattern enables a clean separation of math from boilerplate code.
Also note that for the sake of convenience
we did not worry about automatically inferring the input shape here,
thus we need to specify the number of features throughout.
Do not worry, the `BatchNorm` layer will care of this for us.

class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully-connected layer
    #   or the number of output channels for a convolutional layer.
    # num_dims: 2 for a fully-connected layer and 4 for a convolutional layer.
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter involved in gradient
        # finding and iteration are initialized to 0 and 1 respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # All the variables not involved in gradient finding and iteration are
        # initialized to 0 on the CPU
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # If X is not on the CPU, copy `moving_mean` and `moving_var` to the
        # device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y



### Function Preparation

We will define several functions here. Because all these functions we have used before, so we can use them directly.

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display

import torchvision
from torch.utils import data
from torchvision import transforms

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
            if (i+1) % 10 == 0:
                print('Time Sum:{}, Time Average:{}'.format(timer.sum(), timer.avg()))
            if (i+1) % 50 == 0:
                animator.add(epoch + i/len(train_iter), (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print(f'loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

### Using a Batch Normalization LeNet

To see how to apply `BatchNorm` in context,
below we apply it to a traditional LeNet model.
Recall that BN is typically applied
after the convolutional layers and fully-connected layers
but before the corresponding activation functions.

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(7056, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))



X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)





As before, we will train our network on the Fashion-MNIST dataset.
This code is virtually identical to that when we first trained LeNet.
The main difference is the considerably larger learning rate.

Firstly, we prepare the dataset for training.

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



Here we only train one mini-batch. 
We will use the high-level API `nn.BatchNorm1d` comes with Pytorch in the following experiment.
The training time will be much less. 
We just post a simple example here.

batch_size = 32
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)



optimizer = torch.optim.SGD(net.parameters(), lr=1)
loss = nn.CrossEntropyLoss()
device=try_gpu()

for i, (X, y) in enumerate(train_iter):
    X, y = X.to(device), y.to(device)
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    optimizer.step()
    if i>3:
        break



Let us have a look at the scale parameter `gamma`
and the shift parameter `beta` learned
from the first batch normalization layer.

net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))



### Concise Implementation

Compared with the `BatchNorm` class,
which we just defined ourselves,
we can use the `BatchNorm` class defined in high-level APIs directly.
The code looks virtually identical
to the application our implementation above.

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(7056, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))



Below, we use the same hyper-parameters to train out model.
Note that as usual, the high-level API variant runs much faster
because its code has been compiled to C++/CUDA
while our custom implementation must be interpreted by Python.


The training will take about 20-30 minutes.

lr, num_epochs, batch_size = 1.0, 10, 32
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train_ch6(net, train_iter, test_iter, num_epochs, lr)



### Controversy

Intuitively, batch normalization is thought
to make the optimization landscape smoother.
However, we must be careful to distinguish between
speculative intuitions and true explanations
for the phenomena that we observe when training deep models.
Recall that we do not even know why simpler
deep neural networks (MLPs and conventional CNNs)
generalize well in the first place.
Even with dropout and L2 regularization,
they remain so flexible that their ability to generalize to unseen data
cannot be explained via conventional learning-theoretic generalization guarantees.

In the original paper proposing batch normalization,
the authors, in addition to introducing a powerful and useful tool,
offered an explanation for why it works:
by reducing *internal covariate shift*.
Presumably by *internal covariate shift* the authors
meant something like the intuition expressed above—the
notion that the distribution of activations changes
over the course of training.
However there were two problems with this explanation:
(1) This drift is very different from *covariate shift*,
rendering the name a misnomer.
(2) The explanation offers an under-specified intuition
but leaves the question of *why precisely this technique works*
an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners
use to guide their development of deep neural networks.
However, we believe that it is important
to separate these guiding intuitions
from established scientific fact.
Eventually, when you master this material
and start writing your own research papers
you will want to be clear to delineate
between technical claims and hunches.

Following the success of batch normalization,
its explanation in terms of *internal covariate shift*
has repeatedly surfaced in debates in the technical literature
and broader discourse about how to present machine learning research.
In a memorable speech given while accepting a Test of Time Award
at the 2017 NeurIPS conference,
Ali Rahimi used *internal covariate shift*
as a focal point in an argument likening
the modern practice of deep learning to alchemy.
Subsequently, the example was revisited in detail
in a position paper outlining
troubling trends in machine learning (`Lipton.Steinhardt.2018`).
In the technical literature other authors (`Santurkar.Tsipras.Ilyas.ea.2018`)
have proposed alternative explanations for the success of BN,
some claiming that BN's success comes despite exhibiting behavior
that is in some ways opposite to those claimed in the original paper.

We note that the *internal covariate shift*
is no more worthy of criticism than any of
thousands of similarly vague claims
made every year in the technical ML literature.
Likely, its resonance as a focal point of these debates
owes to its broad recognizability to the target audience.
Batch normalization has proven an indispensable method,
applied in nearly all deployed image classifiers,
earning the paper that introduced the technique
tens of thousands of citations.

### Summary

* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the minibatch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* Batch Normalization has many beneficial side effects, primarily that of regularization. On the other hand, the original motivation of reducing covariate shift seems not to be a valid explanation.

### Exercises

1. Can we remove the fully connected affine transformation before the batch normalization or the bias parameter in convolution computation?
    * Find an equivalent transformation that applies prior to the fully connected layer.
    * Is this reformulation effective. Why (not)?
1. Compare the learning rates for LeNet with and without batch normalization.
    * Plot the decrease in training and test error.
    * What about the region of convergence? How large can you make the learning rate?
1. Do we need Batch Normalization in every layer? Experiment with it?
1. Can you replace Dropout by Batch Normalization? How does the behavior change?
1. Fix the coefficients `beta` and `gamma` , and observe and analyze the results.
1. Review the online documentation for `BatchNorm` to see the other applications for Batch Normalization.
1. Research ideas: think of other normalization transforms that you can apply? Can you apply the probability integral transform? How about a full rank covariance estimate?

[Discussions](https://discuss.d2l.ai/t/84)

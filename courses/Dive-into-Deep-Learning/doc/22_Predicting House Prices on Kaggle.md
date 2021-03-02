## Predicting House Prices on Kaggle

Now that we have introduced some basic tools
for building and training deep networks
and regularizing them with techniques including
dimensionality reduction, weight decay, and dropout,
we are ready to put all this knowledge into practice
by participating in a Kaggle competition.
[Predicting house prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) is a great place to start.
The data is fairly generic and does not exhibit exotic structure
that might require specialized models (as audio or video might).
This dataset, collected by Bart de Cock in 2011 (`De-Cock.2011`),
covers house prices in Ames, IA from the period of 2006-2010.
It is considerably larger than the famous [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) of Harrison and Rubinfeld (1978),
boasting both more examples and more features.

In this section, we will walk you through details of
data preprocessing, model design, and hyperparameter selection.
We hope that through a hands-on approach,
you will gain some intuitions that will guide you
in your career as a data scientist.

### Kaggle

[Kaggle](https://www.kaggle.com) is a popular platform
that hosts machine learning competitions.
Each competition centers on a dataset and many
are sponsored by stakeholders who offer prizes
to the winning solutions.
The platform helps users to interact
via forums and shared code,
fostering both collaboration and competition.
While leaderboard chasing often spirals out of control,
with researchers focusing myopically on pre-processing steps
rather than asking fundamental questions,
there is also tremendous value in the objectivity of a platform
that facillitates direct quantitative comparisons
between competing approaches as well as code sharing
so that everyone can learn what did and did not work.
If you want to participate in a Kaggle competitions,
you will first need to register for an account
(see the following figure).

![Kaggle website](https://doc.shiyanlou.com/courses/2777/246442/b1bd324eeda15a7ee6651b8a1f9cea74-0/wm)

On the House Prices Prediction page, as illustrated
in the following figure,
you can find the dataset (under the "Data" tab),
submit predictions, see your ranking, etc.,
The URL is right here:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![House Price Prediction](https://doc.shiyanlou.com/courses/2777/246442/802126cdea92eca2eec65a0a7ef8683c-0/wm)

### Accessing and Reading the Dataset

Note that the competition data is separated
into training and test sets.
Each record includes the property value of the house
and attributes such as street type, year of construction,
roof type, basement condition, etc.
The features consist of various data types.
For example, the year of construction
is represented by an integer,
the roof type by discrete categorical assignments,
and other features by floating point numbers.
And here is where reality complicates things:
for some examples, some data is altogether missing
with the missing value marked simply as *na*.
The price of each house is included
for the training set only
(it is a competition after all).
We will want to partition the training set
to create a validation set,
but we only get to evaluate our models on the official test set
after uploading predictions to Kaggle.
The "Data" tab on the competition tab
has links to download the data.

To get started, we will read in and process the data
using `pandas`, an [efficient data analysis toolkit](http://pandas.pydata.org/pandas-docs/stable/).

from matplotlib import pyplot as plt
%matplotlib inline
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
import numpy as np



For convenience, we have uploaded the dataset in aliyuncs. 
We can download and cache
the Kaggle housing dataset
using the following script.

!wget -nc "https://labfile.oss.aliyuncs.com/courses/2777/kaggle_house_data.zip"
!unzip -o "kaggle_house_data.zip"



To load the two csv files containing training
and test data respectively we use Pandas.


train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')



The training dataset includes $1460$ examples,
$80$ features, and $1$ label, while the test data
contains $1459$ examples and $80$ features.


print(train_data.shape)
print(test_data.shape)



Let’s take a look at the first $4$ and last $2$ features
as well as the label (SalePrice) from the first $4$ examples:


print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])



We can see that in each example, the first feature is the ID.
This helps the model identify each training example.
While this is convenient, it does not carry
any information for prediction purposes.
Hence, we remove it from the dataset
before feeding the data into the network.


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))



### Data Preprocessing

As stated above, we have a wide variety of data types.
We will need to process the data before we can start modeling.
Let us start with the numerical features.
First, we apply a heuristic,
replacing all missing values
by the corresponding variable's mean.
Then, to put all variables on a common scale,
we rescale them to zero mean and unit variance:

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

To verify that this indeed transforms
our variable such that it has zero mean and unit variance,
note that $E[(x-\mu)/\sigma] = (\mu - \mu)/\sigma = 0$
and that $E[(x-\mu)^2] = \sigma^2$.
Intuitively, we *normalize* the data
for two reasons.
First, it proves convenient for optimization.
Second, because we do not know *a priori*
which features will be relevant,
we do not want to penalize coefficients
assigned to one variable more than on any other.

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)



Next we deal with discrete values.
This includes variables such as 'MSZoning'.
We replace them by a one-hot encoding
in the same way that we previously transformed
multiclass labels into vectors.
For instance, 'MSZoning' assumes the values 'RL' and 'RM'.
These map onto vectors $(1, 0)$ and $(0, 1)$ respectively.
Pandas does this automatically for us.


# `Dummy_na=True` refers to a missing value being a legal eigenvalue, and
# creates an indicative feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape

You can see that this conversion increases
the number of features from 79 to 331.
Finally, via the `values` attribute,
we can extract the NumPy format from the Pandas dataframe
and convert it into MXNet's native tensor
representation for training.


n_train = train_data.shape[0] # 训练集样本个数
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,
                             dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values,
                            dtype=torch.float32).reshape(-1, 1)



### Training

To get started we train a linear model with squared loss.
Not surprisingly, our linear model will not lead
to a competition-winning submission
but it provides a sanity check to see whether
there is meaningful information in the data.
If we cannot do better than random guessing here,
then there might be a good chance
that we have a data processing bug.
And if things work, the linear model will serve as a baseline
giving us some intuition about how close the simple model
gets to the best reported models, giving us a sense
of how much gain we should expect from fancier models.

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net



With house prices, as with stock prices,
we care about relative quantities
more than absolute quantities.
Thus we tend to care more about
the relative error $\frac{y - \hat{y}}{y}$
than about the absolute error $y - \hat{y}$.
For instance, if our prediction is off by USD 100,000
when estimating the price of a house in Rural Ohio,
where the value of a typical house is 125,000 USD,
then we are probably doing a horrible job.
On the other hand, if we err by this amount
in Los Altos Hills, California,
this might represent a stunningly accurate prediction
(there, the median house price exceeds 4 million USD).

One way to address this problem is to
measure the discrepancy in the logarithm of the price estimates.
In fact, this is also the official error metric
used by the competition to measure the quality of submissions.
After all, a small value $\delta$ of $\log y - \log \hat{y}$
translates into $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
This leads to the following loss function:

$$L = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

def log_rmse(net,features,labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),
                                       torch.log(labels))))
    return rmse.item()



Unlike in previous sections, our training functions
will rely on the Adam optimizer
(a slight variant on SGD that we will describe
in greater detail later).
The main appeal of Adam vs vanilla SGD
is that the Adam optimizer,
despite doing no better (and sometimes worse)
given unlimited resources for hyperparameter optimization,
people tend to find that it is significantly less sensitive
to the initial learning rate.

def load_array(data_arrays, batch_size, is_train=True): 
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            outputs = net(X)
            l = loss(outputs,y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls



### $k$-Fold Cross-Validation

If you are reading in a linear fashion,
you might recall that we introduced k-fold cross-validation
in the section where we discussed how to deal
with model selection.
We will put this to good use to select the model design
and to adjust the hyperparameters.
We first need a function that returns
the $i^\mathrm{th}$ fold of the data
in a k-fold cross-validation procedure.
It proceeds by slicing out the $i^\mathrm{th}$ segment
as validation data and returning the rest as training data.
Note that this is not the most efficient way of handling data
and we would definitely do something much smarter
if our dataset was considerably larger.
But this added complexity might obfuscate our code unnecessarily
so we can safely omit it here owing to the simplicity of our problem.

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid



The training and verification error averages are returned
when we train $k$ times in the k-fold cross-validation.


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print(f'fold {i}, train rmse {float(train_ls[-1]):f}, '
              f'valid rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k



### Model Selection

In this example, we pick an untuned set of hyperparameters
and leave it up to the reader to improve the model.
Finding a good choice can take time,
depending on how many variables one optimizes over.
With a large enough dataset,
and the normal sorts of hyperparameters,
k-fold cross-validation tends to be
reasonably resilient against multiple testing.
However, if we try an unreasonably large number of options
we might just get lucky and find that our validation
performance is no longer representative of the true error.

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train rmse: {float(train_l):f}, '
      f'avg valid rmse: {float(valid_l):f}')



Notice that someimes the number of training errors
for a set of hyperparameters can be very low,
even as the number of errors on $k$-fold cross-validation
is considerably higher.
This indicates that we are overfitting.
Throughout training you will want to monitor both numbers.
No overfitting might indicate that our data can support a more powerful model.
Massive overfitting might suggest that we can gain
by incorporating regularization techniques.

###  Predict and Submit

Now that we know what a good choice of hyperparameters should be,
we might as well use all the data to train on it
(rather than just $1-1/k$ of the data
that is used in the cross-validation slices).
The model that we obtain in this way
can then be applied to the test set.
Saving the estimates in a CSV file
will simplify uploading the results to Kaggle.

def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'train rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = net(test_features).detach().numpy()
    # Reformat it for export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)



One nice sanity check is to see
whether the predictions on the test set
resemble those of the k-fold cross-validation process.
If they do, it is time to upload them to Kaggle.
The following code will generate a file called `submission.csv`
(CSV is one of the file formats accepted by Kaggle):


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)



Next, as demonstrated in the following figure,
we can submit our predictions on Kaggle
and see how they compare to the actual house prices (labels)
on the test set.
The steps are quite simple:

* Log in to the Kaggle website and visit the House Price Prediction Competition page.
* Click the “Submit Predictions” or “Late Submission” button (as of this writing, the button is located on the right).
* Click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.
* Click the “Make Submission” button at the bottom of the page to view your results.

![Submitting data to Kaggle](https://doc.shiyanlou.com/courses/2777/246442/f7dff3944be7c68ca2357f168d1394f7-0/wm)

### Summary

* Real data often contains a mix of different data types and needs to be preprocessed.
* Rescaling real-valued data to zero mean and unit variance is a good default. So is replacing missing values with their mean.
* Transforming categorical variables into indicator variables allows us to treat them like vectors.
* We can use k-fold cross validation to select the model and adjust the hyper-parameters.
* Logarithms are useful for relative loss.

### Exercises

1. Submit your predictions for this tutorial to Kaggle. How good are your predictions?
1. Can you improve your model by minimizing the log-price directly? What happens if you try to predict the log price rather than the price?
1. Is it always a good idea to replace missing values by their mean? Hint: can you construct a situation where the values are not missing at random?
1. Find a better representation to deal with missing values. Hint: what happens if you add an indicator variable?
1. Improve the score on Kaggle by tuning the hyperparameters through k-fold cross-validation.
1. Improve the score by improving the model (layers, regularization, dropout).
1. What happens if we do not standardize the continuous numerical features like we have done in this section?

[Discussions](https://discuss.d2l.ai/t/107)

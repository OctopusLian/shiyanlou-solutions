## Documentation

Due to constraints on the length of this book, we cannot possibly introduce every single Pytorch function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the Pytroch API.

### Finding All the Functions and Classes in a Module

In order to know which functions and classes can be called in a module, we
invoke the `dir` function. For instance, we can query all properties in the
module for generating random numbers:

import torch
print(dir(torch.distributions))



Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

### Finding the Usage of Specific Functions and Classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let us explore the usage instructions for tensors' `ones` function.

help(torch.ones)



From the documentation, we can see that the `ones` function creates a new tensor with the specified shape and sets all the elements to the value of 1. Whenever possible, you should run a quick test to confirm your interpretation:


torch.ones(4)



In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content that is almost
identical to `help(list)`, displaying it in a new browser
window. In addition, if we use two question marks, such as
`list??`, the Python code implementing the function will also be
displayed.

torch.ones??



### Summary

* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of an API by calling the `dir` and `help` functions, or `?` and `??` in Jupyter notebooks.

### Exercises

1. Look up the documentation for any function or class in the deep learning framework. Can you also find the documentation on the official website of the framework?

[Discussions](https://discuss.d2l.ai/t/39)

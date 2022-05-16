# %% [markdown]
#  # Introduction to PyTorch's AlexNet model
# 
#  AlexNet is one of the many architectures avaliable in the computer vision sector of artificial intelligence. This model was primarily designed for classification tasks but can be used for a range of other applications, such as feature extraction.
#  The version of AlexNet used in this notebook was trained with ImageNet1k data set which has 1.4 million high-resolution images spread across 1,000 categories.
#  In this notebook, we tested AlexNet. First with an image with a class belonging to the known labels and then with the MNIST dataset so the model predictions could be observed.

# %% [markdown]
#  # Module Imports
# 
#  A module is a part of a python package. Packages are available in various common sources like PyPI.
#  Using import statements, we can make the code in one module, or package, available in our code.
# 
#  PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets.
#  The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors.
#  Additionally, it provides many utilities for efficient serialisation of Tensors and arbitrary types, along with other useful utilities.
#  To use a preferred module in our program, we need to install it in our local machine using `pip` or `conda` installer.
# 
#  The torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
# 
#  In our module import section, we have imported models, transforms, and datasets from torchvision package.
#  The models module contains definitions for the different model (AlexNet, ResNet, GoogLeNet etc) architectures for image classification.
#  Transforms are common image transformations available in torch vision, such as the conversion of a PIL image to a tensor.
#  Instead of building a dataset from the scratch, we can utilise some built-in datasets appropriate for our needs
#  and Torchvision helps us by providing many built-in datasets in the torchvision.datasets module.
# 
#  So all these imports can be loaded as shown in the following code chunk.

# %%

import torch 
from torchvision import models          # Models are subpackage contains definitions for the different model architectures for image classification
from torchvision import transforms      # Transforms are common image transforms
from PIL import Image                   # PIL is the Python Imaging Library which provides the python interpreter with image editing capabilities.
import requests                         # Requests module allows you to send HTTP requests using Python.

from torchvision import datasets



# %% [markdown]
#  # Data and Model Download
# 
#  Raw data can be any unprocessed fact, value, text, sound, or picture that is not being interpreted and analysed.
#  A “model” in machine learning is the output of a machine learning algorithm run on data.
#  A model represents what was learned by a machine learning algorithm.
#  Each model will have its own architecture. The AlexNet model used in our code has eight layers with learnable parameters.
# 
#  A model has weights and biases for each neuron. Each input is multiplied by a weight to indicate the input's relative importance.
#  Bias is then added to the the weighted inputs.
#  An activation function then performs a calculation on the total.
# 
#  A pre-trained mode is a model created by someone else to solve a similar problem. These pre-trained models are usually trained on very large datasets using high-end hardware.
#  Instead of building a model from scratch to solve a similar problem, you can use a model trained on another problem as a starting point. This method allows us to solve a problem quickly by leveraging knowledge previously learned by the network.
#  Some of the popular pre-trained image classification models are AlexNet, ResNet, and GoogleLeNet.
#  We can also create our own model, by manually defining the each of the required layers and then training on a large amount of data.
# 
#  To utilize the pre-trained feature of a model in pytorch,
#  we need to pass the pretrained parameter as True in the model initialization step like shown in the below code snippet.
# 

# %%
AlexNet = models.alexnet(pretrained=True)     # Utilizing pre-trained alexnet model from PyTorch


# %% [markdown]
#  Just call the object name like in the below snippet, to view the details of each layers in the alexnet architecture.

# %%
AlexNet


# %% [markdown]
#  In Pytorch, we can set the model to operate under one of two modes (Train mode and Evaluation mode). By default all the models are initialized to train mode.
#  model.train() sets the modules in the network in training mode.
#  It tells our model that we are currently in the training phase so the model keeps some layers,
#  like dropout, batch-normalization which behaves differently depends on the current phase, active.
#  whereas the model.eval() does the opposite.
#  Therefore, once the model.eval() has been called our model will deactivate such layers so that the model outputs its inference as is expected.
#  Since we are not going to perform any training here, we have directly called the evaluation mode.

# %%
AlexNet.eval()                              # options which will help our model to understand we are in evaluation mode. Also, Equivalent to model.train(False).
                                            # Eval will switch off the requires_grad and dropout and batch normalization

# %% [markdown]
#  Alexnet is trained with the ImageNet dataset, which has 1000 labels.
#  During our evaluation process, alexnet will map our input image into one of these 1000 categories.
#  So, in the below snippet, we fetch the labels from the URL using an HTTP request and store them in a variable called labels.

# %%
url1 ="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(url1).text.split('\n')

print(len(labels))

# %% [markdown]
#  # Input Data
# 
#  The model expects the input to be in a tensor batch. mini-batches of 3-channel RGB images of shape (C x H x W), where C is channels, H (height) and W (width) are expected to be at least 224.
#  Our expected batch shape should be [B x C x H x W]. where B is the size of the batch.
#  The values that make up the image need to be scaled to a range of [0, 1], this is achieved by normalisingthe data mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
#  This mean and standard deviation has been derived from the imagenet dataset and is used here because pretrained model is trained on these data.
# 
#  A model can understand only numeric values. If we have some non numerical data, then we need to convert it into numeric representation through some kind of transformation.

# %%

transform = transforms.Compose([     # Compose class Composes several transforms together.
    transforms.Resize(256),          # Resize the input image to the given size.
    transforms.CenterCrop(224),      # Crops the given image at the center.
    transforms.ToTensor(),           # Convert a PIL Image or numpy.ndarray to tensor. 
    transforms.Normalize(            
        mean=[0.485, 0.456, 0.406],    
        std=[0.229, 0.224, 0.225])
        ])


# %% [markdown]
#  In the below snippet, a sample strawberry image is chosen and loaded in a variable called img.

# %%
#strwaberry
url ="https://learnopencv.com/wp-content/uploads/2021/01/strawberries.jpg"
img = Image.open(requests.get(url,stream=True).raw)     # Performing http request to fetch the image.                                   

img

# %% [markdown]
#  Then our img is passed into our transform object to achieve the required tensor and
#   later this transformed_image object will be used as the input of our `predictImage` method.

# %%
transformed_image = transform(img)                                 # Transforming our image into the required format.


# %% [markdown]
#  # Output Prediction
# 
#  Output of the model will be a confidence score for each of the classes.
#  For example, in AlexNet the output will be confidence scores over Imagenet's 1000 classes.
# 
#  The first step of the prediction process is to create a batch. Then we need to pass the batch to our alexnet model.
#  Now the model will return a vector of numbers that we stored in a variable called output as shown in the below snippet.
#  Followed by that we need to convert this vetcor into probabilities.
#  Using softmax function in pyTorch, we can easily convert a vector of numbers into a vector of probabilities,
#  where the probabilities of each value are proportional to the relative scale of each value in the vector and the sum of all probabilities is equal to one.
# 
#  Finally, with the help of the torch.sort() function, we sorted all elements in the output tensor in descending order.
#  Then using a simple for loop, we iterated through the first five class labels and probabilities and returned a variable called top5.
#  By printing it we will get the top 5 predictions of our model.

# %%
def predictImage(tImage, model):                              # User defined function with two inputs. Tensor image and Model type.
    bImage = torch.unsqueeze(tImage,0)                      # Matching the size of our input with the trained set input size. Also for batch processing.
    #bImage = tImage.unsqueeze(0)                           
    output = model(bImage)
    print((output.size()))
    percentage = torch.nn.functional.softmax(output,dim =1)[0]*100   # Softmax operation is applied to all slices of input along the specified dim, and will rescale them so that the elements lie in the range (0, 1) and sum to 1.  
    _, ids = torch.sort(output, descending=True)                    # torch.sort return the maximum value of all elements in the input tensor in desceding order
    top5 = [(labels[idx],percentage[idx].item()) for idx in ids[0][:5] ]
    print(top5)





# %%

predictImage(transformed_image, AlexNet) # Calling the predictImage method and passing the url and model type. Expecting to print the top 5 matching results.



# %% [markdown]
#  # Testing alexnet with MNIST dataset
# 
#  Alexnet accepts input only in the form of a tensor, so we need to transform the MNIST dataset into tensor.
#  In the next step, we download the MNIST data set onto our local machine.
#  Finally, we are pass a single MNIST data sample into our predictImage method, which will return the top 5 predictions of AlexNet.
#  As the MNIST dataset is in one dimensional, because the images are greyscale, using transform we create 2 more channels by duplicating the original data.
#  This is required because the alexnet requires an RGB image which is 3 dimensional.

# %%
MNIST_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:tensor.repeat(3,1,1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=MNIST_transform,
)

# %% [markdown]
#  First input from our training dataset is fetched and the corresponding numeric value of the data is 5.
#  Then the tensor relvant to the dataset is passed into our predict Image function.

# %%
m_img, num = training_data[0]
num

# %%
predictImage(m_img,AlexNet)

# %% [markdown]
#  Since, we haven't trained the model, with MNIST dataset. Our model predicted it as nematode.

# %%




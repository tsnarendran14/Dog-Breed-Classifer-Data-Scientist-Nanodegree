# Data Scientist Nanodegree Capstone Project - Dog Breed Classifier

## Project Overview

The objective of this project is, if a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. 
We will be using Convolution Neural Network (CNN) since it works very well for images.

## Summary of the project:

We will use deep learning algorithm called Convolution neural network to classify dog breeds. We will use open cv2 to detect humans and dogs from the images. Then we will build our own CNN architecture. The results from this CNN model is not convincing as it gives very bad accuracy. So we will use transfer learning concept. That is, we will use pre-trained model. In our case, we will use ResNet50 model. The accuracy which we get here is around 80 percent. In order to test our application, we will also use a sample of six images to test real world scenario.

## Interesting and difficult things in the project:

It was amazing to see how CNN algorithm works so well in images. There are 133 categories and CNN does so well in predicting those categories. The most difficult part is building my own CNN algorithm. Neural network has so many parameters that it is difficult to tune them. But thanks to udacity’s sample model, which helped me to build my model with greater than one percent accuracy.

## Metrics Used:

For calculating the performance of the model, we will use accuracy. Accuracy improvement is the requirement of the project and also it gives a very clear information about how well we are able to classify dog breeds. For example, for 100 images, if we are able to classify 60 images correctly, then the accuracy is 60%.

Loss Function: The loss function of neural network is categorical crossentropy. It is the most popular loss function used in neural network for multi class classification which uses softmax activation function. The loss function is differentiable. 

## Contents

Step 0: Import Datasets

Step 1: Detect Humans

Step 2: Detect Dogs

Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)

Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

Step 6: Write your Algorithm

Step 7: Test Your Algorithm

## Blog Link - For detailed analysis and explanation of the model

https://medium.com/@tsnarendran14/dog-breed-classifier-with-keras-walk-through-70c9b0a2b17f

### Setup Instructions

1. Clone the repository and navigate to the downloaded folder.
Reference Link ->
				https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.


## Final Outcome
Used Resnet50 model. The accuracy of test dataset is 83 percent, which beats the set benchmark at 60 percent

## Copyrights:
This is a open source project.
For usage guidelines, please refer to https://www.udacity.com/legal

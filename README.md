# NNNumpy
NNNumpy is a simple artifitial neural networks library implemented using numpy (python 3.6). The code is not efficient (when using concovolution layers), but it is very clear.

Now there is one model: sequential neural network.
1. Layers
  - FullyConnected (with one of activation functions)
  - Activation
  - Dropout
  - BatchNormalization
  - Flatten
  - Convolution2D
  - MaxPool2D
2. Activation functions (you can run activations.py file to see graphics)
  - Linear
  - TanH
  - ReLU
  - SoftPlus
  - Sigmoid
  - ArcTan
  - ISRU
  - BentIdentity
  - Sinusoid
  - Gaussian
  - SoftMax
3. Regularizations
  - L1Regularization
  - L2Regularization
  - L1L2Regularization
4. Losses
  - AbsoluteError (mean absolute error)
  - SquaredError (mean squared error)
  - CrossEntropy (categorial)
5. Optimizers
  - SGD
  - NesterovAG (Nesterov accelerated gradient)
  - Adagrad
  - RMSprop
  - Adam
6. Helpers
  - LRScheduler (learning rate scheduler)
  - EarlyStopper
  - ModelSaver

_________________________________________________________________
## Examples
1. Regression (regression_test.py)
Model fitted on noised data (blue and orange dots)
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/regr_test.png?raw=true)
2. Classification (classification_test.py)
It is the dataset
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/classif_dataset.png?raw=true)
And predictions after fitting (~97% accuracy on the whole dataset)
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/classif_result.png?raw=true)
3. MNIST
 You can use a pre-trained models (re-train it) or train your own models.

  - MLP (mnist_mlp_test.py)
  This model after 15 epochs gets to 97.3% validation accuracy.
> NeuralNetwork (7 layers) input_shape: (None, 784)
> no. : Layer name (output_shape)	: description
> 1 : FullyConnected (None, 400)	: ArcTan(-Pi/2,Pi/2)(314000 params)
> 2 : BatchNormalization (None, 400)	: (800 params)
> 3 : Dropout (None, 400)	: None
> 4 : FullyConnected (None, 400)	: ArcTan(-Pi/2,Pi/2)(160400 params)
> 5 : BatchNormalization (None, 400)	: (800 params)
> 6 : Dropout (None, 400)	: None
> 7 : FullyConnected (None, 10)	: SoftMax(probabilities)(4010 params)
> Total params num: 480010
  - CNN (mnist_cnn_test.py)
  And this model after 5 epochs gets to 98.45% validation accuracy.
> NeuralNetwork (9 layers) input_shape: (None, 1, 28, 28)
> no. : Layer name (output_shape)	: description
> 1 : Convolution2D (None, 12, 28, 28)	: 12 3x3 ReLU[0,inf)(120 params)
> 2 : MaxPool2D (None, 12, 14, 14)	: 2x2
> 3 : Convolution2D (None, 24, 14, 14)	: 24 3x3 ReLU[0,inf)(2616 params)
> 4 : MaxPool2D (None, 24, 7, 7)	: 2x2
> 5 : Convolution2D (None, 48, 7, 7)	: 48 3x3 ReLU[0,inf)(10416 params)
> 6 : Flatten (None, 2352)	: None
> 7 : BatchNormalization (None, 2352)	: (4704 params)
> 8 : FullyConnected (None, 1176)	: ReLU[0,inf)(2767128 params)
> 9 : FullyConnected (None, 10)	: SoftMax(probabilities)(11770 params)
> Total params num: 2796754

4. Cifar-10
There are also pre-trained models that you can check (re-train) or train your own models.
  - MLP
  This model gets to 51.13% validation accuracy after 10 epochs.
> NeuralNetwork (6 layers) input_shape: (None, 3072)
> no. : Layer name (output_shape)	: description
> 1 : FullyConnected (None, 350)	: ArcTan(-Pi/2,Pi/2)(1075550 params)
> 2 : BatchNormalization (None, 350)	: (700 params)
> 3 : FullyConnected (None, 300)	: ArcTan(-Pi/2,Pi/2)(105300 params)
> 4 : BatchNormalization (None, 300)	: (600 params)
> 5 : FullyConnected (None, 200)	: ArcTan(-Pi/2,Pi/2)(60200 params)
> 6 : FullyConnected (None, 10)	: SoftMax(probabilities)(2010 params)
> Total params num: 1244360

  - CNN
  And this model gets to 64.4% validation accuracy after 12 epochs.
 > NeuralNetwork (18 layers) input_shape: (None, 3, 32, 32)
> no. : Layer name (output_shape)	: description
> 1 : Convolution2D (None, 12, 32, 32)	: 12 3x3 Linear(-inf,inf)(336 params)
> 2 : BatchNormalization (None, 12, 32, 32)	: (24576 params)
> 3 : Activation (None, 12, 32, 32)	: ReLU[0,inf)
> 4 : Convolution2D (None, 12, 32, 32)	: 12 3x3 Linear(-inf,inf)(1308 params)
> 5 : BatchNormalization (None, 12, 32, 32)	: (24576 params)
> 6 : Activation (None, 12, 32, 32)	: ReLU[0,inf)
> 7 : MaxPool2D (None, 12, 16, 16)	: 2x2
> 8 : Convolution2D (None, 24, 16, 16)	: 24 3x3 Linear(-inf,inf)(2616 params)
> 9 : BatchNormalization (None, 24, 16, 16)	: (12288 params)
> 10 : Activation (None, 24, 16, 16)	: ReLU[0,inf)
> 11 : Convolution2D (None, 24, 16, 16)	: 24 3x3 Linear(-inf,inf)(5208 params)
> 12 : BatchNormalization (None, 24, 16, 16)	: (12288 params)
> 13 : Activation (None, 24, 16, 16)	: ReLU[0,inf)
> 14 : MaxPool2D (None, 24, 8, 8)	: 2x2# NNNumpy
NNNumpy is a simple artifitial neural networks library implemented using numpy (python 3.6). The code is not efficient (when using concovolution layers), but it is very clear.

Now there is one model: sequential neural network.
1. Layers
  - FullyConnected (with one of activation functions)
  - Activation
  - Dropout
  - BatchNormalization
  - Flatten
  - Convolution2D
  - MaxPool2D
2. Activation functions (you can run activations.py file to see graphics)
  - Linear
  - TanH
  - ReLU
  - SoftPlus
  - Sigmoid
  - ArcTan
  - ISRU
  - BentIdentity
  - Sinusoid
  - Gaussian
  - SoftMax
3. Regularizations
  - L1Regularization
  - L2Regularization
  - L1L2Regularization
4. Losses
  - AbsoluteError (mean absolute error)
  - SquaredError (mean squared error)
  - CrossEntropy (categorial)
5. Optimizers
  - SGD
  - NesterovAG (Nesterov accelerated gradient)
  - Adagrad
  - RMSprop
  - Adam
6. Helpers
  - LRScheduler (learning rate scheduler)
  - EarlyStopper
  - ModelSaver

_________________________________________________________________
## Examples
1. Regression (regression_test.py)
Model fitted on noised data (blue and orange dots)
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/regr_test.png?raw=true)
2. Classification (classification_test.py)
It is the dataset
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/classif_dataset.png?raw=true)
And predictions after fitting (~97% accuracy on the whole dataset)
![alt text](https://github.com/zBlur/NNNumpy/blob/master/NN_tests/classif_result.png?raw=true)
3. MNIST
 You can use a pre-trained models (re-train it) or train your own models.

  - MLP (mnist_mlp_test.py)
  This model after 15 epochs gets to 97.3% validation accuracy.
> NeuralNetwork (7 layers) input_shape: (None, 784)
no. : Layer name (output_shape)	: description
1 : FullyConnected (None, 400)	: ArcTan(-Pi/2,Pi/2)(314000 params)
2 : BatchNormalization (None, 400)	: (800 params)
3 : Dropout (None, 400)	: None
4 : FullyConnected (None, 400)	: ArcTan(-Pi/2,Pi/2)(160400 params)
5 : BatchNormalization (None, 400)	: (800 params)
6 : Dropout (None, 400)	: None
7 : FullyConnected (None, 10)	: SoftMax(probabilities)(4010 params)
Total params num: 480010
  - CNN (mnist_cnn_test.py)
  And this model after 5 epochs gets to 98.45% validation accuracy.
> NeuralNetwork (9 layers) input_shape: (None, 1, 28, 28)
no. : Layer name (output_shape)	: description
1 : Convolution2D (None, 12, 28, 28)	: 12 3x3 ReLU[0,inf)(120 params)
2 : MaxPool2D (None, 12, 14, 14)	: 2x2
3 : Convolution2D (None, 24, 14, 14)	: 24 3x3 ReLU[0,inf)(2616 params)
4 : MaxPool2D (None, 24, 7, 7)	: 2x2
5 : Convolution2D (None, 48, 7, 7)	: 48 3x3 ReLU[0,inf)(10416 params)
6 : Flatten (None, 2352)	: None
7 : BatchNormalization (None, 2352)	: (4704 params)
8 : FullyConnected (None, 1176)	: ReLU[0,inf)(2767128 params)
9 : FullyConnected (None, 10)	: SoftMax(probabilities)(11770 params)
Total params num: 2796754

4. Cifar-10
There are also pre-trained models that you can check (re-train) or train your own models.
  - MLP
  This model gets to 51.13% validation accuracy after 10 epochs.
> NeuralNetwork (6 layers) input_shape: (None, 3072)
no. : Layer name (output_shape)	: description
1 : FullyConnected (None, 350)	: ArcTan(-Pi/2,Pi/2)(1075550 params)
2 : BatchNormalization (None, 350)	: (700 params)
3 : FullyConnected (None, 300)	: ArcTan(-Pi/2,Pi/2)(105300 params)
4 : BatchNormalization (None, 300)	: (600 params)
5 : FullyConnected (None, 200)	: ArcTan(-Pi/2,Pi/2)(60200 params)
6 : FullyConnected (None, 10)	: SoftMax(probabilities)(2010 params)
Total params num: 1244360

  - CNN
  And this model gets to 64.4% validation accuracy after 12 epochs.
> NeuralNetwork (18 layers) input_shape: (None, 3, 32, 32)
no. : Layer name (output_shape)	: description
1 : Convolution2D (None, 12, 32, 32)	: 12 3x3 Linear(-inf,inf)(336 params)
2 : BatchNormalization (None, 12, 32, 32)	: (24576 params)
3 : Activation (None, 12, 32, 32)	: ReLU[0,inf)
4 : Convolution2D (None, 12, 32, 32)	: 12 3x3 Linear(-inf,inf)(1308 params)
5 : BatchNormalization (None, 12, 32, 32)	: (24576 params)
6 : Activation (None, 12, 32, 32)	: ReLU[0,inf)
7 : MaxPool2D (None, 12, 16, 16)	: 2x2
8 : Convolution2D (None, 24, 16, 16)	: 24 3x3 Linear(-inf,inf)(2616 params)
9 : BatchNormalization (None, 24, 16, 16)	: (12288 params)
10 : Activation (None, 24, 16, 16)	: ReLU[0,inf)
11 : Convolution2D (None, 24, 16, 16)	: 24 3x3 Linear(-inf,inf)(5208 params)
12 : BatchNormalization (None, 24, 16, 16)	: (12288 params)
13 : Activation (None, 24, 16, 16)	: ReLU[0,inf)
14 : MaxPool2D (None, 24, 8, 8)	: 2x2
15 : Flatten (None, 1536)	: None
16 : BatchNormalization (None, 1536)	: (3072 params)
17 : FullyConnected (None, 384)	: ReLU[0,inf)(590208 params)
18 : FullyConnected (None, 10)	: SoftMax(probabilities)(3850 params)
Total params num: 680326




> 15 : Flatten (None, 1536)	: None
> 16 : BatchNormalization (None, 1536)	: (3072 params)
> 17 : FullyConnected (None, 384)	: ReLU[0,inf)(590208 params)
> 18 : FullyConnected (None, 10)	: SoftMax(probabilities)(3850 params)
> Total params num: 680326



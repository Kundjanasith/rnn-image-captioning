# AI Image Captioning

We use a pre-trained Convolutional Neural Network to generate embeddings for the images in our dataset.

The embeddings are then used to train a Recurrent Neural Network in order to generate captions in natural language for our images.

You can find the paper [here](https://arxiv.org/pdf/1411.4555).

Here's a plot of the training loss after the 12th epoch:

![alt text](https://github.com/dpstart/rnn-image-captioning/blob/master/training_loss.png)

## How to run

### Notebook version

You can use __run.ipynb__ to run the model.

In order to skip the feature generation for the CNN, you can download the pickle files [here](https://github.com/dpstart/rnn-image-captioning/releases/tag/v0.1).

It is also possible to skip the training of the RNN by loading the pretrained weights that you can find in this repo:

* *pretrained_weights_14_epoch.tar.gz* : Weights after 14 epochs of training.
* *pretrained_weights_16_epoch.tar.gz* : Weights after 16 epochs of training.

Once you unzip the files, you can use: `saver.restore(s, os.path.abspath("weights_<epoch_number>"))` before applying the model in order to use the weights.

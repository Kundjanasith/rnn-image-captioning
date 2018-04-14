# AI Image Captioning

We use a pre-trained Convolutional Neural Network to generate embeddings for the images in our dataset.

The embeddings are then used to train a Recurrent Neural Network in order to generate captions in natural language for our images.

You can find the paper [here](https://arxiv.org/pdf/1411.4555).

Here's a plot of the training loss after the 12th epoch:

<p align="center">
  <img src="https://github.com/dpstart/rnn-image-captioning/blob/master/training_loss.png" width="350"/>
</p>


## How to run

### Notebook version

You can use __run.ipynb__ to run the model.

In order to skip the feature generation for the CNN, you can download the pickle files [here](https://github.com/dpstart/rnn-image-captioning/releases/tag/v0.1).

It is also possible to skip the training of the RNN by loading the pretrained weights that you can find in this repo:

* *final_weights.zip* : Weights after 16 epochs of training.

Once you unzip the files, you can use: `saver.restore(s, os.path.abspath("weights_<epoch_number>"))` before applying the model in order to use the weights.


## Examples

<p align="center">
  <img src="https://github.com/dpstart/rnn-image-captioning/blob/master/images/example_1.jpeg" width="350"/>
</p>

<p align="center">
  <img src="https://github.com/dpstart/rnn-image-captioning/blob/master/images/example_2.jpeg" width="350"/>
</p>

<p align="center">
  <img src="https://github.com/dpstart/rnn-image-captioning/blob/master/images/example_3.jpeg" width="350"/>
</p>

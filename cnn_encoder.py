import sys
import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
L = keras.layers
K = keras.backend

"""
Dataset:
train images http://msvocds.blob.core.windows.net/coco2014/train2014.zip
validation images http://msvocds.blob.core.windows.net/coco2014/val2014.zip
captions for both train and validation http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
"""

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model


if __name__ == '__main__':
    # load pre-trained model
    K.clear_session()
    encoder, preprocess_for_model = get_cnn_encoder()

    # extract train features
    train_img_embeds, train_img_fns = utils.apply_model(
        "train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(train_img_embeds, "train_img_embeds.pickle")
    utils.save_pickle(train_img_fns, "train_img_fns.pickle")

    # extract validation features
    val_img_embeds, val_img_fns = utils.apply_model(
        "val2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
    utils.save_pickle(val_img_embeds, "val_img_embeds.pickle")
    utils.save_pickle(val_img_fns, "val_img_fns.pickle")

    # sample images for learners
    def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
        np.random.seed(seed)
        with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
            sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
            for zInfo in sampled:
                fout.writestr(zInfo, fin.read(zInfo))
                
    sample_zip("train2014.zip", "train2014_sample.zip")
    sample_zip("val2014.zip", "val2014_sample.zip")

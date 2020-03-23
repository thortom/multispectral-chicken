#################################################################################################
# Code in some part from:                                                                       #
#    https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb #
#################################################################################################
# The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py).
#
# The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.

# As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process.
import numpy as np
import tensorflow as tf
import IPython
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
print(f'TensorFlow version is {tf.__version__}')

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(image, label):
    input_image = tf.image.resize(image, (128, 128), method='nearest')
    input_mask = tf.image.resize(label, (128, 128), method='nearest')

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(image, label):
    input_image = tf.image.resize(image, (128, 128), method='nearest')
    input_mask = tf.image.resize(label, (128, 128), method='nearest')

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    classes_predicted = np.unique(display_list[-1], return_counts=True)
    print(len(classes_predicted[0]))
    if len(classes_predicted[0]) < 10:
        print("Predicted classes")
        print(classes_predicted[0])
        print("Occurrences of each:")
        print(classes_predicted[1])
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
#         print(display_list[i].shape)
#         print(f"{title[i]} counts as {len(np.unique(display_list[i], return_counts=True)[0])}")
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def unet_model(output_channels, up_stack, down_stack):

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])

def run(X_train, Y_train, X_test, Y_test, output_chanels, epochs=3, steps_per_epoch=1, clear_output=True):
    TRAIN_LENGTH = 1
    BATCH_SIZE = 1
    BUFFER_SIZE = 1 # What is this?

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    train = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test_dataset.map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    for image, mask in train.take(1):
        sample_image, sample_mask = image[:,:,2:5], mask
    display([sample_image, sample_mask])

    ####################
    # Define the model #

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # ## Train the model
    # Now, all that is left to do is to compile and train the model. The loss being used here is losses.sparse_categorical_crossentropy. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and losses.sparse_categorical_crossentropy is the recommended loss for such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.

    model = unet_model(output_chanels, up_stack, down_stack)

    # Investigate this more
    # https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    opt = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.09)
    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-06)
    # optimizer='adam', loss='sparse_categorical_crossentropy'
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', # categorical_crossentropy
                  metrics=['accuracy'])

    plt.figure()
    tf.keras.utils.plot_model(model, show_shapes=True)
    plt.show()


    # Let's try out the model to see what it predicts before training.

    show_predictions(model, test_dataset)

    VAL_SUBSPLITS = 1
    VALIDATION_STEPS = 1
    model_history = model.fit(train_dataset, epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset)#,
                            #   callbacks=[DisplayCallback()])

    if clear_output:
        IPython.display.clear_output(wait=True)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epoch_list = range(epochs)

    plt.figure()
    plt.plot(epoch_list, loss, 'r', label='Training loss')
    plt.plot(epoch_list, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
#     plt.ylim([0, 1])
    plt.legend()
    plt.show()

    # Show how the model preforms on the test dataset
    show_predictions(model, test_dataset)

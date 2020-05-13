import tensorflow as tf
import tensorflow.keras.backend as K

# # Similar code here: https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html      
# #          and here: https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/cost.html
# # Code from: https://github.com/keras-team/keras/issues/9395
# # Ref: salehi17, "Twersky loss function for image segmentation using 2D FCDN"
# # -> the score is computed for each class separately and then summed
# # alpha=beta=0.5 : dice coefficient
# # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# # alpha+beta=1   : produces set of F*-scores
# # implemented by E. Moebel, 06/04/18
# def tversky_loss(y_true, y_pred):
#     alpha = 0.5
#     beta  = 0.5
    
#     ones = K.ones(K.shape(y_true))
#     p0 = y_pred      # probability that voxels are class i
#     p1 = ones-y_pred # probability that voxels are not class i
#     g0 = y_true
#     g1 = ones-y_true
    
#     num = K.sum(p0*g0, (0,1,2))
#     den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
#     T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
#     Ncl = K.cast(K.shape(y_true)[-1], 'float32')
#     return Ncl - T

# Binary Twersky loss
# From: https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html
def tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-10, only_contaminant=False):
    """ Tversky loss function.

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    def join_classes(y):
        # The expected input contains the labels {0: 'Belt', 1: 'Chicken', 2: 'Contaminant'}
        if only_contaminant: # Ignores all except contaminants. That is 'Contaminant' becomes 1, 'Not Contaminant' becomes 0
            return y[...,2:]
        else:                # This ignores the background. This is enough to classify all well.
            # Does this make sense?... It does at least work rather well
            return y[...,1:] # Here the 'Chicken' becomes 0 and 'Contaminant' becomes 1
    
    y_true = K.flatten(join_classes(y_true)) # Flatten binary-one-hot-encoding to binary array
    y_pred = K.flatten(join_classes(y_pred))
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    # tf.print(y_pred)
    
    return 1 - answer

# Focal Loss
# From: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
def focal_loss(gamma=2., alpha=.25):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(value=y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(value=y_pred, dtype=tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(input_tensor=fl, axis=1)
        return tf.reduce_mean(input_tensor=reduced_fl)
    return focal_loss_fixed

# import tensorflow_addons as tfa
# fl = tfa.losses.SigmoidFocalCrossEntropy()

# TODO: Test this implementation of IOU out
# from keras import backend as K
def iou_coef(y_true, y_pred, smooth=1):
    # Code from: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou
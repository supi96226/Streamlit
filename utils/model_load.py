import pickle
import tensorflow as tf
from tensorflow.keras.models import model_from_json


def binary_focal_loss(gamma=2., alpha=0.25):
    """
    Custom loss function for binary classification using focal loss.

    :param gamma: Focusing parameter.
    :param alpha: Balance parameter.
    :return: A focal loss function.
    """

    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        loss = -alpha_t * tf.keras.backend.pow(1. - p_t, gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(loss)
    return focal_loss_fixed


def load_model_from_pickle(pickle_file):
    """
    Load the saved model from a pickle file, and compile it with model architecture and weights.

    :param pickle_file: Path to the pickle file containing the model.
    :return: Compiled Keras model.
    Eg. file path: 'data/best_model_final.pkl'
    """
    
    with open(pickle_file, 'rb') as f:
        model_package = pickle.load(f)

    model_architecture = model_package['architecture']
    model_weights = model_package['weights']

    model = model_from_json(model_architecture)
    model.set_weights(model_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=binary_focal_loss(gamma=2, alpha=0.6),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    print("Model successfully loaded from .pkl and compiled.")
    return model


if __name__ == "__main__":
    model = load_model_from_pickle('data/best_model_final.pkl')
    print(model.summary())
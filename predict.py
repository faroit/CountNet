import numpy as np
import argparse
import os
import keras
import sklearn
import librosa
from keras import backend as K


eps = np.finfo(np.float).eps


def class_mae(y_true, y_pred):
    return K.mean(
        K.abs(
            K.argmax(y_pred, axis=-1) - K.argmax(y_true, axis=-1)
        ),
        axis=-1
    )


def load_scaler():
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']
    return scaler


def load_model(model_name):
    path = os.path.join('models', model_name + '.h5')
    return keras.models.load_model(
        path,
        custom_objects={
            'class_mae': class_mae,
            'exp': K.exp
        }
    )


def count(audio, model, scaler):
    # compute STFT
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    # apply global (featurewise) standardization to mean1, var0
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:500, :]

    # apply l2 normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X[np.newaxis, ...]

    if len(model.input_shape) == 4:
        X = X[:, np.newaxis, ...]

    ys = model.predict(X, verbose=0)
    return np.argmax(ys, axis=1)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )

    parser.add_argument(
        'audio',
        help='audio file (samplerate 16 kHz) of 5 seconds duration',
        nargs='+',
    )

    parser.add_argument(
        '--model', default='CRNN',
        help='model name'
    )

    parser.add_argument('--print-summary', action='store_true')

    args = parser.parse_args()

    # load model
    model = load_model(args.model)

    if args.print_summary:
        # print model configuration
        model.summary()

    # load standardisation parameters
    scaler = load_scaler()

    for f in args.audio:
        # compute audio
        audio = librosa.load(f, sr=16000)[0]
        estimate = count(audio, model, scaler)
        print("Speaker Count Estimate:", f, estimate)

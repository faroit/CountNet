import numpy as np
import soundfile as sf
import argparse
import os
import keras
import sklearn
import librosa

eps = np.finfo(np.float).eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )

    parser.add_argument(
        'audio',
        help='audio file (16 kHz) of 5 seconds duration'
    )

    parser.add_argument(
        '--model', default='RNN',
        help='model name'
    )

    args = parser.parse_args()

    # load model
    model = keras.models.load_model(
        os.path.join('models', args.model + '.h5')
    )

    # print model configuration
    model.summary()
    # save as svg file
    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    # compute audio
    audio, rate = sf.read(args.audio, always_2d=True)

    # downmix to mono
    audio = np.mean(audio, axis=1)

    # compute STFT
    X = np.abs(librosa.stft(audio, n_fft=400, hop_length=160)).T

    # apply standardization
    X = scaler.transform(X)

    # cut to input shape length (500 frames x 201 STFT bins)
    X = X[:model.input_shape[-2], :]

    # apply normalization
    Theta = np.linalg.norm(X, axis=1) + eps
    X /= np.mean(Theta)

    # add sample dimension
    X = X.reshape(model.input_shape[1:])
    Xs = X[np.newaxis, ...]

    # predict output
    ys = model.predict(Xs, verbose=0)
    print("Speaker Count Estimate: ", np.argmax(ys, axis=1)[0])

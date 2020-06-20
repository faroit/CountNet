import numpy as np
import soundfile as sf
import argparse
import os
import keras
import sklearn
import glob
import predict
import json
from keras import backend as K

import tqdm

eps = np.finfo(np.float).eps


def mae(y, p):
    return np.mean([abs(a - b) for a, b in zip(p, y)])


def mae_by_count(y, p):
    diffs = []
    for c in range(0, int(np.max(y)) + 1):
        ind = np.where(y == c)
        diff = mae(y[ind], np.round(p[ind]))
        diffs.append(diff)

    return diffs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load keras model and predict speaker count'
    )
    parser.add_argument(
        'root',
        help='root dir to evaluation data set'
    )

    parser.add_argument(
        '--model', default='CRNN',
        help='model name'
    )

    args = parser.parse_args()

    # load model
    model = keras.models.load_model(
        os.path.join('models', args.model + '.h5'),
        custom_objects={
            'class_mae': predict.class_mae,
            'exp': K.exp
        }
    )


    # print model configuration
    model.summary()

    # load standardisation parameters
    scaler = sklearn.preprocessing.StandardScaler()
    with np.load(os.path.join("models", 'scaler.npz')) as data:
        scaler.mean_ = data['arr_0']
        scaler.scale_ = data['arr_1']

    input_files = glob.glob(os.path.join(
        args.root, 'test', '*.wav'
    ))

    y_trues = []
    y_preds = []

    for input_file in tqdm.tqdm(input_files):

        metadata_file = os.path.splitext(
            os.path.basename(input_file)
        )[0] + ".json"
        metadata_path = os.path.join(args.root, 'test', metadata_file)

        with open(metadata_path) as data_file:
            data = json.load(data_file)
            # add ground truth
            y_trues.append(len(data))

        # compute audio
        audio, rate = sf.read(input_file, always_2d=True)

        # downmix to mono
        audio = np.mean(audio, axis=1)

        count = predict.count(audio, model, scaler)
        # add prediction
        y_preds.append(count)

    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)


mae_k = mae_by_count(y_trues, y_preds)
print("MAE per Count: ", {k: v for k, v in enumerate(mae_k)})
print("Mean MAE", mae(y_trues, y_preds))

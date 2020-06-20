# Speaker Count Estimation using Deep Neural Networks

<img width="400" align="right" alt="screen shot 2017-11-21 at 12 35 28" src="https://user-images.githubusercontent.com/72940/33071669-be6c35b2-cebc-11e7-8822-9b998ad1ea09.png">

_CountNet_ is a deep learning model to estimate the number of concurrent speakers from single channel mixtures is a very challenging task that is a mandatory ﬁrst step to address any realistic “cocktail-party” scenario. It has various audio-based applications such as blind source separation, speaker diarisation, and audio surveillance.

This repo provides pre-trained models.

## Publications

### 2019: IEEE/ACM Transactions on Audio, Speech, and Language Processing

* __Title__: CountNet: Estimating the Number of Concurrent Speakers Using Supervised Learning
 Speaker Count Estimation
* __Authors__: [Fabian-Robert Stöter](https://faroit.com), Soumitro Chakrabarty, Bernd Edler, Emanuël
 A. P. Habets
* __Preprint__: [HAL](https://hal-lirmm.ccsd.cnrs.fr/lirmm-02010805)
* __Proceedings__: [IEEE](https://ieeexplore.ieee.org/document/8506601) (paywall)

### 2018: ICASSP 

* __Title__: Classification vs. Regression in Supervised Learning for Single Channel
 Speaker Count Estimation
* __Authors__: [Fabian-Robert Stöter](https://faroit.com), Soumitro Chakrabarty, Bernd Edler, Emanuël
 A. P. Habets
* __Preprint__: [arXiv 1712.04555](http://arxiv.org/abs/1712.04555)
* __Proceedings__: [IEEE](https://ieeexplore.ieee.org/document/8462159) (paywall)

## Demos

A demo video is provided on the [accompanying website](https://www.audiolabs-erlangen.de/resources/2017-CountNet).

## Usage

This repository provides the [keras](https://keras.io/) model to be used from Python to infer count estimates. The preprocessing dependes on librosa and scikit-learn. Not that the provided model is trained on 16 kHz samples of 5 seconds duration. 

### Docker

[Docker](https://www.docker.com/) makes it easy to reproduce the results and install all requirements. If you have docker installed, run the following steps to predict a count from the provided test sample.

* Build the docker image: `docker build -t countnet .`
* Predict from example: `docker run -i countnet python predict.py --model CRNN examples/5_speakers.wav`

### Manual Installation 

To install the requirements using Anaconda Python, run

`conda env create -f env.yml`

You can now run the command line script and process wav files using the pre-trained model `CRNN` (best peformance).

`python predict.py examples/5_speakers.wav --model CRNN`

## Reproduce Paper Results using the LibriCount Dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1216072.svg)](https://doi.org/10.5281/zenodo.1216072)

The full test dataset is available for download on [Zenodo](https://doi.org/10.5281/zenodo.1216072).

### LibriCount10 0dB Dataset

The dataset contains a simulated cocktail party environment of [0..10] speakers, mixed with 0dB SNR from random utterances of different speakers from the [LibriSpeech](http://www.openslr.org/12/) `CleanTest` dataset. 

For each recording we provide the ground truth number of speakers within the file name, where `k` in, `k_uniquefile.wav` is the maximum number of concurrent speakers with the 5 seconds of recording.

All recordings are of 5s durations. For each unique recording, we provide the audio wave file (16bits, 16kHz, mono) and an annotation `json` file with the same name as the recording.

### Metadata

In the annotation file we provide information about the speakers sex, their unique speaker_id, and vocal activity within the mixture recording in samples. Note that these were automatically generated using [a voice activity detection method](https://github.com/wiseman/py-webrtcvad).

In the following example a speaker count of 3 speakers is the ground truth.

```json
[
	{
		"sex": "F", 
		"activity": [[0, 51076], [51396, 55400], [56681, 80000]], 
		"speaker_id": 1221
	}, 
	{
		"sex": "F", 
		"activity": [[0, 51877], [56201, 80000]], 
		"speaker_id": 3570
	}, 
	{
		"sex": "M", 
		"activity": [[0, 15681], [16161, 68213], [73498, 80000]], 
		"speaker_id": 5105
	}
]
```

### Running evaluation

```python eval.py ~/path/to/LibriCount10-0dB --model CRNN``` outputs the _mean absolute error_ per class and averaged.

### Pretrained models

| Name     | Number of Parameters | MAE on test set |
|----------|----------------------|-----------------|
| `RNN`    | 0.31M                | 0.38            |
| `F-CRNN` | 0.06M                | 0.36            |
| `CRNN`   | 0.35M                | __0.27__        |


## FAQ

#### Is it possible to convert the model to run on a modern version of keras with tensorflow backend?

Yes, its possible. But I was unable to get identical results when converting model. I tried this [guide](https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa) but it still didn't help to get to the same performance compared to keras 1.2.2 and theano.

## License

MIT

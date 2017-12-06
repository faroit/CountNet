# Speaker Count Estimation using Deep Neural Networks

<img width="400" align="right" alt="screen shot 2017-11-21 at 12 35 28" src="https://user-images.githubusercontent.com/72940/33071669-be6c35b2-cebc-11e7-8822-9b998ad1ea09.png">

Estimating the number of concurrent speakers from single channel mixtures is a very challenging task that is a mandatory ﬁrst step to address any realistic “cocktail-party” scenario. It has various audio-based applications such as blind source separation, speaker diarisation, and audio surveillance. Building upon powerful machine learning methodology and the possibility to generate large amounts of learning data, Deep Neural Network (DNN) architectures are well suited to directly estimate speaker counts.

## Model

<img width="400" alt="screen shot 2017-11-21 at 12 35 28" src="https://user-images.githubusercontent.com/72940/33072095-60d1929c-cebe-11e7-91de-1dff3fc50bde.png">

In this work a recurrent neural network was trained to generate speaker count estimates for 0 to 10 speakers. The model uses three Bi-LSTM layers inspired by a model for singing voice separation by [Leglaive15](https://hal.archives-ouvertes.fr/hal-01110035).


## Demos

A demo video is provided on the [accompanying website](https://www.audiolabs-erlangen.de/resources/2017-CountNet).

## Usage

This repository provides the [keras](https://keras.io/) model to be used from Python to infer count estimates. The preprocessing dependes on librosa and scikit-learn. Not that the provided model is trained on 16 kHz samples of 5 seconds duration. 

### Docker

[Docker](https://www.docker.com/) makes it easy to reproduce the results and install all requirements. If you have docker installed, run the following steps to predict a count from the provided test sample.

* Build the docker image: `docker build -t countnet .`
* Predict from example: `docker run -i countnet python predict_audio.py examples/5_speakers.wav`

### Manual Installation 

Make sure you have Python 3.6, `libsndfile` and `libhdf5` installed on your system (e.g. through Anaconda). To install the requirements run

`pip install -r requirements.txt`

You can now run the command line script and process wav files

`python predict_audio.py examples/5_speakers.wav`

## Reproduce Paper Results

We will provide the full test dataset soon.

## License

MIT

FROM conda/miniconda3

RUN apt update && apt install -y g++

# Copy requirements.txt and run pip first so that changes to the application
# code do not require a rebuild of the entire image
COPY requirements.txt /app/
RUN conda update conda && \
    conda install "keras<2.4" "numpy<2" "scikit-learn<0.23" && \
    conda install -c conda-forge librosa theano

ADD . /app
WORKDIR /app

VOLUME /data

ENV KERAS_BACKEND=theano

FROM conda/miniconda3

RUN apt-get update && \
    apt-get install -y libsndfile1

ADD . /app
WORKDIR /app

VOLUME /data

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

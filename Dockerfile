FROM conda/miniconda3

RUN apt-get update && \
    apt-get install -y libsndfile1

ADD . /app
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

FROM python:3.7

WORKDIR /core

COPY ./requirements.txt /core/requirements.txt

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install -r /core/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /core/
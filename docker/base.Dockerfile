FROM python:3-bullseye

RUN apt-get update && \
    apt-get upgrade -y

ADD ./ /opt/vectorboss/

WORKDIR /opt/vectorboss/

RUN pip install -r ./requirements.txt

FROM python:3-bullseye

ADD ./ /opt/vectorboss/

WORKDIR /opt/vectorboss/

RUN pip install -r ./requirements.txt

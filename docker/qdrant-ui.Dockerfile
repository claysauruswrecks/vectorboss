FROM node:lts-bullseye

RUN apt-get update && \
    apt-get upgrade -y

RUN git clone https://github.com/qdrant/qdrant-web-ui.git /opt/qdrant-ui/

WORKDIR /opt/qdrant-ui/

RUN sed -i 's/localhost/qdrant/g' /opt/qdrant-ui/src/common/axios.js

RUN npm install

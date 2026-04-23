FROM node:current-slim

WORKDIR /usr/src/app

COPY . .

RUN apt update
RUN apt-get -y install npm
RUN apt-get -y install python3
RUN apt-get -y install make

RUN npm ci --omit=dev

CMD [ "node", "index.ts" ]
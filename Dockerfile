FROM python:3.8

RUN python3 --version
RUN pip3 -q install pip

RUN mkdir src
WORKDIR src/
COPY . .
EXPOSE 8888

RUN pip3 install -r requirements.txt



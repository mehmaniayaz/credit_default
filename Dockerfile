FROM python:3.8

RUN python3 --version
RUN pip3 -q install pip

RUN mkdir src
WORKDIR src/
COPY . .

RUN pip3 install -r requirements.txt
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]



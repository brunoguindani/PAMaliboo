## First of all, cd into the folder containing this file.
## Build the image with:
#  docker build -t pamaliboo .
## Run container (bash) with:
#  docker run --cpus 4 --name pam --rm -v $(pwd):/pamaliboo -it pamaliboo
## Remove root permissions from "outputs" folder (if Docker was run by root):
#  chmod -R a+rw outputs

FROM python:3.10
ENV MY_DIR=/pamaliboo
WORKDIR ${MY_DIR}
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
COPY . .

CMD bash

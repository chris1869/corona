FROM python:3.7-alpine

WORKDIR /app
COPY requirements.txt /app

RUN apk add --no-cache python3-dev libstdc++ && \
    apk add --no-cache g++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h && \
    pip3 install -r requirements.txt

COPY *.py /app/

CMD ["python3", "worker.py"]

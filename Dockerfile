FROM jjanzic/docker-python3-opencv
COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r requirements.txt
# ENTRYPOINT ["python3"]
# CMD ["flask_api.py"]
CMD gunicorn flask_api:app
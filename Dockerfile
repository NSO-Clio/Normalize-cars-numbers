FROM python:3.9
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
RUN pip install -r requirements.txt
COPY main.py main.py
COPY model.py model.py
COPY encoder_weights.pth encoder_weights.pth
COPY unet_model.pth unet_model.pth
CMD ["python", "main.py"]

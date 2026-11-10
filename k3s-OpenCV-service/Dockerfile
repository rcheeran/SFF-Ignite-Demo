FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY k3s-OpenCV-service/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY k3s-vision-service/app.py /app/base_app.py
COPY k3s-vision-service/vision.py /app/base_vision.py
COPY k3s-OpenCV-service/app.py k3s-OpenCV-service/vision.py ./
COPY k3s-OpenCV-service/test_app.py /app/tests/test_app.py

EXPOSE 8094
CMD ["python", "app.py"]

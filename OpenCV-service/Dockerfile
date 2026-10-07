FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY OpenCV-service/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY vision-service/app.py /app/base_app.py
COPY vision-service/vision.py /app/base_vision.py
RUN sed -i 's/OWL-ViT/OpenCV + Foundry YOLO/g' /app/base_app.py
COPY OpenCV-service/app.py OpenCV-service/vision.py ./
COPY OpenCV-service/test_vision.py /app/tests/test_vision.py
COPY vision-service/test_app.py /app/tests/test_app.py

EXPOSE 8094
CMD ["python", "app.py"]

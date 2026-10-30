FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libglib2.0-0 libx11-6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /models \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', '/models/hand_landmarker.task')"

COPY app.py hand_logic.py ./
RUN python -c "import mediapipe as mp; options=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path='/models/hand_landmarker.task')); landmarker=mp.tasks.vision.HandLandmarker.create_from_options(options); landmarker.close()"

EXPOSE 8095
CMD ["python", "app.py"]

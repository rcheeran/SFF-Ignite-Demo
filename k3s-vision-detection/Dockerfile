FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py vision.py detection_client.py ./
COPY test_app.py test_vision.py test_client.py /app/tests/

EXPOSE 8096
CMD ["python", "app.py"]

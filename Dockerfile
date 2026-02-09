FROM python:3.10-slim


ENV PIP_NO_CACHE_DIR=1

WORKDIR /app


COPY pyproject.toml .
COPY src ./src


RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu


RUN pip install .


COPY . .


CMD ["python", "-m", "src.predict"]


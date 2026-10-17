FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir "mistralai-workflows[mistralai]>=3.10" "httpx>=0.27" "tzdata>=2024.1"

COPY src ./src

CMD ["python", "src/entrypoints/worker.py"]

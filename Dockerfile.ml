FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System deps for common Python packages and curl for healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip "poetry==1.8.4" poetry-plugin-export \
    && poetry export --with ml --without-hashes --format requirements.txt --output requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY ml ./ml
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "ml.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.12-slim as builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    cargo \
    python3-dev \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY IE-app/pyproject.toml IE-app/poetry.lock /app/

RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-interaction --no-ansi

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

COPY IE-app /app

EXPOSE 8000

RUN useradd -m myuser
USER myuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

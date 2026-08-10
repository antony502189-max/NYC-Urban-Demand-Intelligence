FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && pip install ".[ml,api]"

USER app
EXPOSE 8000

CMD ["uvicorn", "nyc_demand.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

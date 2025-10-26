FROM agrigorev/zoomcamp-model:2025
WORKDIR /code

RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
RUN uv sync --no-dev || (echo "No lockfile, continuing" && uv sync --no-dev)

COPY app.py .
COPY pipeline_v1.bin .

EXPOSE 9696
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0

WORKDIR /app

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Copy the source code BEFORE running uv sync so the package can be built
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Now sync and install the package
RUN uv sync --frozen --no-dev

# Ensure the ledger data directory exists
RUN mkdir -p data

EXPOSE 8000

CMD ["uv", "run", "python", "scripts/console.py"]

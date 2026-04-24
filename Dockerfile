# using uv binary as official image faster than pip install uv 
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
# forcing the architecture to Linux/AMD64 for Mac compatibility
FROM --platform=linux/amd64 python:3.11-slim

# copying uv binary into production image
COPY --from=uv_bin /uv /uv/bin/uv

WORKDIR /workspace

# installing system dependencies for XGBoost/Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copying configuration files to cache dependencies
COPY pyproject.toml uv.lock ./

# installing dependencies using uv
RUN /uv/bin/uv sync --frozen --no-dev --system

# copying entire project
COPY . .

# setting pythonpath for the project
ENV PYTHONPATH=/workspace

# documentation : telling docker that the container is designed to listen on port 8080
# using port 8080 for cloud run
EXPOSE 8080

# using uvicorn to spin up server to listen
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
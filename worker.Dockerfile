# Stage 1: Build Rust worker binary
FROM rust:1.85-bookworm AS rust-builder
WORKDIR /build
COPY worker-rust/Cargo.toml worker-rust/Cargo.lock* ./
COPY worker-rust/src/ src/
RUN cargo build --release

# Stage 2: Runtime with Python + Rust binary
FROM python:3.11-slim-bookworm

# Install system deps for gymnasium
RUN apt-get update && apt-get install -y --no-install-recommends \
    swig build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy Rust worker binary
COPY --from=rust-builder /build/target/release/gaia-worker /usr/local/bin/gaia-worker

# Install Python dependencies
WORKDIR /app
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy experiment code
COPY worker/ ./

# Environment
ENV GAIA_SERVER=http://server:7434 \
    GAIA_TOKEN=changeme \
    GAIA_WORKER_NAME=docker-cpu-worker

# Entrypoint: start the Rust worker pointing at local Python experiments
ENTRYPOINT ["gaia-worker"]
CMD ["--experiments-dir", "/app"]

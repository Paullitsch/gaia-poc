FROM rust:1.85-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY static/ static/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/gaia-server /usr/local/bin/gaia-server
ENV GAIA_PORT=7434 GAIA_TOKEN=changeme GAIA_DATA_DIR=/data
EXPOSE 7434
VOLUME /data
ENTRYPOINT ["gaia-server"]

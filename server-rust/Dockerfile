FROM rust:1.85-bookworm AS builder
WORKDIR /build
# Copy protocol crate first (dependency)
COPY protocol/ protocol/
# Copy server crate
COPY server-rust/Cargo.toml server-rust/Cargo.lock* server-rust/
COPY server-rust/src/ server-rust/src/
COPY server-rust/static/ server-rust/static/
WORKDIR /build/server-rust
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/server-rust/target/release/gaia-server /usr/local/bin/gaia-server
ENV GAIA_PORT=7434 GAIA_TOKEN=changeme GAIA_DATA_DIR=/data
EXPOSE 7434 7435
VOLUME /data
ENTRYPOINT ["gaia-server"]

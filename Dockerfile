FROM rust:1.88-slim-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin image-depth-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/image-depth-server /usr/local/bin/image-depth-server
COPY model_quantized.onnx /app/model_quantized.onnx
EXPOSE 7860
CMD ["/usr/local/bin/image-depth-server", "--port", "7860", "--model", "/app/model_quantized.onnx", "--cache-dir", "/app/cache"]

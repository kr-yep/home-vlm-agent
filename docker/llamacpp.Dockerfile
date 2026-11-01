# syntax=docker/dockerfile:1.7
# llama.cpp b11869 (2026-10-24) from its GitHub release, checksum-pinned.
# ghcr only publishes floating server-cuda tags for recent builds, and a
# floating tag would not respect the one-week release cooldown.
FROM ubuntu:24.04
# archive.ubuntu.com drops out from this network now and then; the Taiwan mirror does not.
RUN sed -i 's|http://archive.ubuntu.com|http://tw.archive.ubuntu.com|g; s|http://security.ubuntu.com|http://tw.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:554f4006f4e2456c4dcc0482a207230602758d3f5a700f55c8754c58f4088a29 \
    https://github.com/ggml-org/llama.cpp/releases/download/b11869/llama-b11869-bin-ubuntu-cuda-13.4-x64.tar.gz /tmp/llama.tgz
ADD --checksum=sha256:d879e8e8070bcfcfd4f8bc01664e52787b2253409b5be4c3dd6cfd7e1409f385 \
    https://github.com/ggml-org/llama.cpp/releases/download/b11869/cudart-llama-b11869-bin-ubuntu-cuda-13.4-x64.tar.gz /tmp/cudart.tgz
RUN mkdir -p /opt/llama && tar -xzf /tmp/llama.tgz -C /opt/llama --strip-components=1 \
 && tar -xzf /tmp/cudart.tgz -C /opt/llama --strip-components=1 && rm /tmp/*.tgz
ENV LD_LIBRARY_PATH=/opt/llama PATH=/opt/llama:$PATH
ENTRYPOINT ["llama-server"]

FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 as builder
ENV DEBIAN_FRONTEND=noninteractive

# Install deps
RUN apt update && apt install librust-alsa-sys-dev curl build-essential -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Compile deps for cache efficiency
WORKDIR /usr/src
RUN cargo new musicgpt
WORKDIR /usr/src/musicgpt
COPY Cargo.toml Cargo.lock ./
RUN cargo build --features cuda --release

# Compile the code.
COPY . .
RUN touch src/main.rs # <- this updates the file date in the filesystem, and cargo no longer incorrectly caches the old src/main.rs
RUN cargo build --features cuda --release

# bundle the shared libraries in lib/ folder
RUN mkdir lib
RUN \
for file in target/release/*.so*; do \
  name=${file##*/}; \
  cp $(readlink $file) lib/$name; \
done

FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 as runner
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install librust-alsa-sys-dev ca-certificates -y

COPY --from=builder /usr/src/musicgpt/lib/* /usr/lib64/
COPY --from=builder /usr/src/musicgpt/target/release/musicgpt /usr/bin/

ENV LD_LIBRARY_PATH="/usr/lib64:${LD_LIBRARY_PATH}"

# https://stackoverflow.com/questions/32727594/how-to-pass-arguments-to-shell-script-through-docker-run
ENTRYPOINT ["/bin/sh", "-c", "musicgpt \"$@\"", "--"]


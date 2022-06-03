FROM debian:bullseye as build
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-build,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential

WORKDIR /app

COPY wheels/ /wheels/

# Install into virtual environment
COPY requirements.txt ./

RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    python3 -m venv .venv && \
    .venv/bin/pip3 install --upgrade pip && \
    find . -name 'requirements.txt' -type f | \
    xargs printf '-r %s\n' | \
    xargs .venv/bin/pip3 install --no-cache-dir -f /wheels -f 'https://synesthesiam.github.io/prebuilt-apps'

# -----------------------------------------------------------------------------

FROM debian:bullseye-slim as run
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Dir::Cache var/cache/apt/${TARGETARCH}${TARGETVARIANT};" > /etc/apt/apt.conf.d/01cache

RUN --mount=type=cache,id=apt-run,target=/var/cache/apt \
    mkdir -p /var/cache/apt/${TARGETARCH}${TARGETVARIANT}/archives/partial && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        python3 alsa-utils libopenblas-base perl

WORKDIR /app

# Copy virtual environment
COPY --from=build /app/ ./

COPY LICENSE README.md ./

COPY data/vad_silero/ ./data/vad_silero/
COPY data/wake_precise/ ./data/wake_precise/

COPY data/stt_fsticuffs/en-us/ ./data/stt_fsticuffs/en-us/
COPY data/stt_fsticuffs/kaldi/steps/ ./data/stt_fsticuffs/kaldi/steps/
COPY data/stt_fsticuffs/kaldi/utils/ ./data/stt_fsticuffs/kaldi/utils/
COPY data/stt_fsticuffs/kaldi/${TARGETARCH}${TARGETVARIANT}/ ./data/stt_fsticuffs/kaldi/${TARGETARCH}${TARGETVARIANT}/
RUN cd data/stt_fsticuffs/kaldi/ && \
    if [ -d 'amd64' ]; then ln -s 'amd64' 'x86_64'; fi && \
    if [ -d 'arm64' ]; then ln -s 'arm64' 'aarch64'; fi

# Copy code
COPY rhasspy_junior/ ./rhasspy_junior/
COPY junior.toml ./

COPY scripts/ ./scripts/
COPY docker/run.sh ./

# Clean up
RUN rm -f /etc/apt/apt.conf.d/01cache

ENTRYPOINT ["/app/run.sh"]

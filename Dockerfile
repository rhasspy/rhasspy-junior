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

# Install into virtual environment
COPY requirements.txt ./

RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \
    python3 -m venv .venv && \
    .venv/bin/pip3 install --upgrade pip && \
    find . -name 'requirements.txt' -type f | \
    xargs printf '-r %s\n' | \
    xargs .venv/bin/pip3 install --no-cache-dir -f 'https://synesthesiam.github.io/prebuilt-apps'

# Copy code
COPY LICENSE ./

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

COPY --from=build /app/ ./
COPY scripts/run.sh ./scripts/

# Clean up
RUN rm -f /etc/apt/apt.conf.d/01cache

ENTRYPOINT ["./run.sh"]

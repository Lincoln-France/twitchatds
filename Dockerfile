
ARG BASE_CONTAINER=python:3.7-buster
FROM $BASE_CONTAINER

ARG PUID
ARG PGID
ARG USERNAME=lincoln
ARG WORKDIR=/package
ARG VERSION=0.1.0
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

RUN echo "Image target platform details :: "
RUN echo "TARGETOS:      $TARGETOS"
RUN echo "TARGETARCH:    $TARGETARCH"
RUN echo "TARGETVARIANT: $TARGETVARIANT"

WORKDIR $WORKDIR

RUN pip install --upgrade pip

ENV SHELL="/bin/bash"
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN groupadd --gid $PGID $USERNAME \
    && useradd --uid $PUID --gid $PGID -m $USERNAME \
    && chown -R $PUID:$PGID $WORKDIR

COPY requirements/* requirements/

RUN pip install -r requirements/prod.txt
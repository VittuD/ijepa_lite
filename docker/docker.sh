#!/usr/bin/env bash
set -euo pipefail

REGISTRY="eidos-service.di.unito.it/vitturini"
PROJECT="ijepa-lite"

BASE_TAG="${REGISTRY}/${PROJECT}:base"
DEV_TAG="${REGISTRY}/${PROJECT}:dev"
TRAIN_TAG="${REGISTRY}/${PROJECT}:train"

docker build -t "${BASE_TAG}"  -f docker/Dockerfile_base  .
docker push "${BASE_TAG}"

docker build -t "${DEV_TAG}"   -f docker/Dockerfile_dev   .
docker push "${DEV_TAG}"

docker build -t "${TRAIN_TAG}" -f docker/Dockerfile_train .
docker push "${TRAIN_TAG}"

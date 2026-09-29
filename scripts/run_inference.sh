#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPOSITORY_DIR}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run containerized inference" >&2
    exit 127
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "the Docker Compose v2 plugin is required" >&2
    exit 127
fi

mkdir -p outputs
USE_GPU=${REID_USE_GPU:-0}
BUILD_IMAGE=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            USE_GPU=1
            shift
            ;;
        --build)
            BUILD_IMAGE=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

BUILD_ARGUMENT=""
if [ "${BUILD_IMAGE}" = "1" ]; then
    BUILD_ARGUMENT="--build"
fi

if [ "${USE_GPU}" = "1" ]; then
    docker compose \
        -f docker-compose.yml \
        run --rm ${BUILD_ARGUMENT} --no-deps reid-api \
        python -m scripts.infer_score_optimized \
        --release-config /app/configs/score_optimized_speed.json \
        --weights-dir /app/weights \
        --images-dir /app/dataset/images \
        --query-csv /app/dataset/test_query.csv \
        --gallery-csv /app/dataset/test_gallery.csv \
        --output-dir /app/outputs/score_optimized_speed \
        "$@"
else
    echo "score-optimized inference requires CUDA; rerun with --gpu" >&2
    exit 2
fi

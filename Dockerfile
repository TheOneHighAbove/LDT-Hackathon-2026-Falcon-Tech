# syntax=docker/dockerfile:1.7

# CUDA is required by the score-optimized release profile.
# CUDA 12.1 is intentionally used here: it is supported by the CUDA 12.2
# driver declared for the judging host. Newer CUDA runtimes are not guaranteed
# to start on that driver.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime@sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067
FROM ${PYTORCH_IMAGE}

LABEL org.opencontainers.image.title="Falcon Tech Vehicle ReID" \
      org.opencontainers.image.description="Offline score-optimized vehicle retrieval"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build may access PyPI/the official PyTorch wheel index. No package install or
# model download is performed after the image has been built.
COPY requirements-inference.txt /app/requirements-inference.txt
# Application dependencies are installed into the base image's isolated
# Conda environment. Its build-only Ninja wheel has invalid platform metadata
# under current pip and is not used by the TorchScript-only runtime.
RUN python -m pip install --no-cache-dir --requirement /app/requirements-inference.txt && \
    python -m pip uninstall --yes ninja && \
    python -m pip check && \
    python -c "import cv2, lightgbm, numpy, pandas, scipy, sklearn, torch, torchvision; assert torch.__version__ == '2.5.1+cu121'; assert torchvision.__version__ == '0.20.1+cu121'; assert numpy.__version__ == '2.3.5'; assert pandas.__version__ == '2.3.3'; assert scipy.__version__ == '1.16.3'; assert sklearn.__version__ == '1.8.0'; assert lightgbm.__version__ == '4.6.0'; assert cv2.__version__ == '5.0.0'"

# Copy only the audited runtime closure. Training/probe code and web extras are
# intentionally absent from the release image.
COPY src/__init__.py \
     src/data.py \
     src/input_geometry.py \
     src/release_features.py \
     src/release_io.py \
     src/release_models.py \
     src/release_reranking.py \
     src/score_calibration.py \
     src/utils.py \
     /app/src/
COPY scripts/infer_score_optimized.py /app/scripts/infer_score_optimized.py
COPY configs/score_optimized_speed.json /app/configs/score_optimized_speed.json
COPY LICENSE THIRD_PARTY_NOTICES.md /app/
COPY weights/release/convnext_global_parts.ts \
     weights/release/osnet_loss_branch_global_parts.ts \
     weights/release/dinov2_vehicle_cls_tokens.ts \
     /app/weights/release/
COPY weights/dino_token_cross_top50.pt \
     weights/strict_family_gnn_smoothap.pt \
     weights/dino_top25_verifier.pt \
     weights/family_lambdarank_appearance_lbs.joblib \
     weights/dino_patch_matcher.joblib \
     weights/dino_patch_matcher_10x10_lbs.joblib \
     weights/modern_gallery_linker_lossbranch.joblib \
     /app/weights/

# Fail the build if the explicit speed-profile allowlist exceeds the hard cap.
RUN test "$(find /app/weights -type f -printf '%s\n' | awk '{s+=$1} END {print s}')" -lt 2000000000 && \
    python -m py_compile /app/scripts/infer_score_optimized.py /app/src/release_*.py && \
    python -m scripts.infer_score_optimized --help >/dev/null && \
    mkdir -p /app/outputs /tmp/reid-cache

# The runtime is forced offline before Python imports the application.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    XDG_CACHE_HOME=/tmp/reid-cache \
    DEVICE=cuda

STOPSIGNAL SIGTERM

CMD ["python", "-m", "scripts.infer_score_optimized", "--release-config", "/app/configs/score_optimized_speed.json", "--weights-dir", "/app/weights", "--images-dir", "/app/dataset/images", "--query-csv", "/app/dataset/test_query.csv", "--gallery-csv", "/app/dataset/test_gallery.csv", "--output-dir", "/app/outputs/score_optimized_speed"]

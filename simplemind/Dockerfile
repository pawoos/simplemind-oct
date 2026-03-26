FROM mambaorg/micromamba:2.3.0-cuda12.1.1-ubuntu22.04
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba create --yes --file /tmp/env.yaml

RUN mkdir -p /home/mambauser/.cache/huggingface \
    && chmod -R 777 /home/mambauser/.cache
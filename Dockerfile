FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip curl git \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM NIGHTLY directly (required for gemma-4-E4B / qwen3.5 model support).
# Indexes/strategy match the official vLLM Gemma 4 recipe (CUDA 12.9 nightly + cu129 torch):
#   https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
# Nightly is hardcoded (not a build ARG) because RunPod's GitHub build does not pass --build-arg.
# DeepGEMM compile removed: it needs a long source build (RunPod has a 30-min build limit and no
# GPU at build time) and is disabled by default (VLLM_USE_DEEP_GEMM=0), so the target models don't need it.
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system -U vllm --pre \
        --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4 \
    # DeepGEMM kernels are not bundled in this nightly image (removed to fit RunPod's build limits);
    # keep disabled. Do not set VLLM_USE_DEEP_GEMM=1 unless you also add the DeepGEMM build back.
    VLLM_USE_DEEP_GEMM=0

ENV PYTHONPATH="/:/vllm-workspace"

# Bleeding-edge transformers from source — gemma-4 / qwen3.5 model definitions land here before
# any PyPI release. Installed LAST so it wins over the transformers pulled by requirements.txt.
RUN uv pip install --system -U git+https://github.com/huggingface/transformers.git

# Bake the official vLLM Gemma 4 tool-calling chat template so agentic tool calls render
# correctly (the model's default template does not emit tool definitions). Enable it at
# runtime with CUSTOM_CHAT_TEMPLATE=/templates/tool_chat_template_gemma4.jinja
ADD https://raw.githubusercontent.com/vllm-project/vllm/main/examples/tool_chat_template_gemma4.jinja /templates/tool_chat_template_gemma4.jinja

COPY src /src
RUN chmod +x /src/start.sh
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]

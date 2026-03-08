FROM lmsysorg/sglang:latest

# Install the latest Transformers from GitHub (includes Qwen3.5 MoE model type)
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    accelerate \
    huggingface_hub \
    hf_transfer \
    sentencepiece \
    tiktoken \
    protobuf

# Install uv package manager
RUN curl -Ls https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /sgl-workspace

# Install worker dependencies
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# Copy source files
COPY handler.py engine.py utils.py download_model.py test_input.json ./
COPY public/ ./public/

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
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true \
    # Use bfloat16 by default — native dtype for Qwen3.5 and optimal on L40S
    DTYPE=bfloat16 \
    # CUDA memory tuning
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # NCCL tuning for multi-GPU
    NCCL_P2P_DISABLE=0 \
    NCCL_IB_DISABLE=0

# Model download script execution
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
        python3 download_model.py; \
    fi

CMD ["python3", "handler.py"]

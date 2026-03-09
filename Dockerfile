# Use stable CUDA 12.6 image as base to guarantee driver compatibility across
# RunPod hosts. Then upgrade SGLang from main branch to get Qwen3.5 MoE support
# (merged in sglang PR#18489). The #subdirectory=python suffix is required
# because sglang is a monorepo and pyproject.toml lives under python/.
FROM lmsysorg/sglang:v0.5.3rc1-cu126

# Upgrade SGLang to main branch (Qwen3.5 MoE support, hybrid attention, etc.)
RUN pip install --no-cache-dir --upgrade \
    'git+https://github.com/sgl-project/sglang.git#subdirectory=python'

# Patch SGLang model_config.py to treat Qwen3.5 text-only distilled models as
# non-multimodal. Qwen3_5MoeForConditionalGeneration is in SGLang's
# multimodal_model_archs list, so SGLang calls AutoProcessor.from_pretrained()
# for ALL models with this architecture — even text-only distilled variants that
# have no processor_config.json. The fix mirrors the existing mm_disabled_models
# mechanism already used for Gemma3/Llama4/Step3VL.
# Root cause confirmed in: sglang/srt/configs/model_config.py (mm_disabled_models)
#                          sglang/srt/managers/tokenizer_manager.py (_get_processor_wrapper)
RUN python3 - << 'PATCH'
import inspect
import sglang.srt.configs.model_config as mc

filepath = inspect.getfile(mc)
with open(filepath, "r") as f:
    content = f.read()

old_snippet = '"Gemma3ForConditionalGeneration",'
new_snippet = (
    '"Gemma3ForConditionalGeneration",\n'
    '                "Qwen3_5MoeForConditionalGeneration",\n'
    '                "Qwen3_5ForConditionalGeneration",'
)

if "Qwen3_5MoeForConditionalGeneration" not in content.split("mm_disabled_models")[1].split("]")[0]:
    content = content.replace(old_snippet, new_snippet, 1)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Patched {filepath}: added Qwen3_5MoeForConditionalGeneration to mm_disabled_models")
else:
    print("Already patched, skipping")
PATCH

# PyTorch 2.9.1 (pulled in by SGLang main) has a known bug with CuDNN < 9.15.
# Install the required CuDNN version as recommended by SGLang's own check.
# Reference: https://github.com/pytorch/pytorch/issues/168167
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29

# Install additional ML dependencies
# (transformers, sentencepiece, tiktoken are already in the base image)
RUN pip install --no-cache-dir \
    accelerate \
    huggingface_hub \
    hf_transfer \
    protobuf

# Set working directory
WORKDIR /sgl-workspace

# Install worker dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

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
    # Silence the PyTorch 2.9.1/CuDNN compatibility check — we install 9.16.0.29 above
    SGLANG_DISABLE_CUDNN_CHECK=1 \
    # Use bfloat16 by default — native dtype for Qwen3.5 and optimal on L40S
    DTYPE=bfloat16 \
    # CUDA memory tuning (PYTORCH_CUDA_ALLOC_CONF was deprecated, use PYTORCH_ALLOC_CONF)
    PYTORCH_ALLOC_CONF=expandable_segments:True \
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

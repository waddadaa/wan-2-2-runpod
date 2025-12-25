"""
WAN 2.2 Image-to-Video (I2V) Handler for RunPod Serverless
Model: Wan2.2-I2V-A14B
License: Apache 2.0 (Commercial use allowed)

ALL PARAMETERS EXPOSED
"""

import os
import sys
import gc
import torch
import base64
import tempfile
import traceback
import logging
from typing import Dict, Any
from PIL import Image
from io import BytesIO

import runpod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix safetensors "device cuda:0 is invalid" error (Rust CUDA bug workaround)
import safetensors.torch
_original_load_file = safetensors.torch.load_file

def _patched_load_file(filename, device="cpu"):
    result = _original_load_file(filename, device="cpu")
    if device != "cpu" and torch.cuda.is_available():
        target = torch.device("cuda:0")
        result = {k: v.to(target) for k, v in result.items()}
    return result

safetensors.torch.load_file = _patched_load_file
logger.info("Patched safetensors to load via CPU")

sys.path.insert(0, "/app/Wan2.2")

MODEL = None

# Model configuration
MODEL_NAME = "Wan2.2-I2V-A14B"
HF_REPO_ID = "Wan-AI/Wan2.2-I2V-A14B"
WAN_CONFIG_KEY = "i2v-A14B"


def ensure_model_downloaded(model_dir: str, model_name: str, hf_repo_id: str) -> str:
    """Check if model exists, download from HuggingFace if not."""
    ckpt_dir = os.path.join(model_dir, model_name)

    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0:
        logger.info(f"Model found at {ckpt_dir}")
        return ckpt_dir

    logger.info("=" * 60)
    logger.info(f"Model not found at {ckpt_dir}")
    logger.info(f"Downloading {model_name} from HuggingFace...")
    logger.info(f"Repo: {hf_repo_id}")
    logger.info("This may take a while (~50GB download)")
    logger.info("=" * 60)

    try:
        from huggingface_hub import snapshot_download
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"Model downloaded successfully to {ckpt_dir}")
        return ckpt_dir
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Failed to download model from {hf_repo_id}: {e}")


def load_model():
    """Load I2V model, downloading from HuggingFace if needed."""
    global MODEL
    if MODEL is not None:
        return MODEL

    # Initialize CUDA before loading model
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        _ = torch.zeros(1).cuda()
        logger.info(f"CUDA initialized: {torch.cuda.get_device_name(0)}")

    model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models")
    ckpt_dir = ensure_model_downloaded(model_dir, MODEL_NAME, HF_REPO_ID)

    logger.info("=" * 60)
    logger.info("Loading WAN 2.2 I2V model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[WAN_CONFIG_KEY]

        MODEL = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("I2V model loaded successfully!")
        logger.info("=" * 60)
        return MODEL
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def decode_base64_data(base64_string: str) -> bytes:
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    return base64.b64decode(base64_string)


def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_resolution(size: str) -> tuple:
    if not size:
        return (1280, 720)
    if "*" in size:
        parts = size.split("*")
    elif "x" in size.lower():
        parts = size.lower().split("x")
    else:
        return (1280, 720)
    return (int(parts[0]), int(parts[1]))


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image from bytes and return it with dimensions."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return img


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 Image-to-Video Handler - ALL PARAMETERS

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    image (str):
        Base64 encoded input image.
        Supports PNG, JPG, WEBP formats.
        Can include data URI prefix: "data:image/png;base64,..."

    ═══════════════════════════════════════════════════════════════════════════
    GENERATION PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str, default: ""):
        Text description of desired motion/action.
        Example: "The woman turns her head and smiles"
        If empty, model generates natural motion automatically.

    negative_prompt (str, default: "..."):
        What to avoid in generation.

    size (str, default: "1280x720"):
        Output resolution. Format: "WIDTHxHEIGHT" or "WIDTH*HEIGHT"
        Supported: "1280x720" (720P), "832x480" (480P)

    num_frames / frame_num (int, default: 81):
        Number of frames. Must be 4n+1 format.
        Range: 5-257

    ═══════════════════════════════════════════════════════════════════════════
    SAMPLING PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    sample_steps / num_inference_steps (int, default: 50):
        Denoising steps. Range: 10-150

    sample_guide_scale / guidance_scale (float, default: 5.0):
        CFG scale. Range: 1.0-20.0

    sample_shift / flow_shift (float, default: 5.0):
        Flow matching shift. Range: 1.0-10.0

    sample_solver (str, default: "unipc"):
        Solver: "unipc" or "dpm++"

    ═══════════════════════════════════════════════════════════════════════════
    PROMPT EXTENSION
    ═══════════════════════════════════════════════════════════════════════════

    use_prompt_extend (bool, default: false):
        Enable prompt enhancement.
        For I2V, uses vision-language model to describe image + motion.

    prompt_extend_method (str, default: "local_qwen"):
        Options: "dashscope", "local_qwen"

    prompt_extend_model (str, default: "Qwen/Qwen2.5-VL-7B-Instruct"):
        Vision-language model for I2V prompt extension.

    prompt_extend_target_lang (str, default: "en"):
        Options: "en", "zh"

    ═══════════════════════════════════════════════════════════════════════════
    OUTPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    seed / base_seed (int, default: random):
        Random seed. Use -1 for random.

    fps (int, default: 16):
        Output FPS. Range: 8-60

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "width": 1280,
        "height": 720,
        "num_frames": 81,
        "fps": 16,
        "seed": 12345,
        "input_size": [1920, 1080],
        "prompt_extended": "..." (if use_prompt_extend=true)
    }
    """
    job_input = job.get("input", {})

    try:
        # =====================================================================
        # REQUIRED: IMAGE
        # =====================================================================
        image_data = job_input.get("image", "")
        if not image_data:
            return {"error": "image is required (base64 encoded)"}

        image_bytes = decode_base64_data(image_data)
        img = load_image_from_bytes(image_bytes)
        input_width, input_height = img.size

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, low resolution, "
            "static, no motion, frozen, ugly, bad anatomy, deformed"
        )

        # For I2V, we use max_area - the output follows input image aspect ratio
        size = job_input.get("size", "1280*720")
        width, height = parse_resolution(size)
        max_area = width * height

        num_frames = job_input.get("num_frames", job_input.get("frame_num", 81))
        num_frames = int(num_frames)
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = max(5, min(257, num_frames))

        # =====================================================================
        # SAMPLING PARAMETERS
        # =====================================================================
        sample_steps = int(job_input.get("sample_steps", job_input.get("num_inference_steps", 50)))
        sample_guide_scale = float(job_input.get("sample_guide_scale", job_input.get("guidance_scale", 5.0)))
        sample_shift = float(job_input.get("sample_shift", job_input.get("flow_shift", 5.0)))
        sample_solver = job_input.get("sample_solver", "unipc")

        # =====================================================================
        # PROMPT EXTENSION
        # =====================================================================
        use_prompt_extend = job_input.get("use_prompt_extend", False)
        prompt_extend_method = job_input.get("prompt_extend_method", "local_qwen")
        prompt_extend_model = job_input.get("prompt_extend_model", "Qwen/Qwen2.5-VL-7B-Instruct")
        prompt_extend_target_lang = job_input.get("prompt_extend_target_lang", "en")

        # =====================================================================
        # OUTPUT PARAMETERS
        # =====================================================================
        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        fps = max(8, min(60, int(job_input.get("fps", 16))))

        # =====================================================================
        # GENERATE
        # =====================================================================
        logger.info(f"I2V: {width}x{height}, {num_frames}f, input={input_width}x{input_height}")
        logger.info(f"Sampling: steps={sample_steps}, cfg={sample_guide_scale}, shift={sample_shift}")

        pipeline = load_model()

        # Prompt extension for I2V
        extended_prompt = None
        if use_prompt_extend:
            try:
                from wan.utils.prompt_extend import QwenPromptExpander
                prompt_expander = QwenPromptExpander(
                    model_name=prompt_extend_model,
                    task="i2v-A14B",
                    is_vl=True,
                    device=0
                )
                prompt_output = prompt_expander(
                    prompt,
                    image=img,
                    tar_lang=prompt_extend_target_lang,
                    seed=seed
                )
                if prompt_output.status:
                    extended_prompt = prompt_output.prompt
                    logger.info(f"Extended: {extended_prompt[:100]}...")
                else:
                    logger.warning(f"Prompt extension failed: {prompt_output.message}")
            except Exception as e:
                logger.warning(f"Prompt extension failed: {e}")

        final_prompt = extended_prompt if extended_prompt else prompt

        with torch.inference_mode():
            video = pipeline.generate(
                final_prompt,
                img,
                max_area=max_area,
                frame_num=num_frames,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                n_prompt=negative_prompt,
                seed=seed,
                offload_model=True
            )

        output_path = tempfile.mktemp(suffix=".mp4")
        from wan.utils.utils import save_video
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        video_base64 = encode_video_base64(output_path)

        # Calculate actual output dimensions (follows input aspect ratio)
        aspect_ratio = input_height / input_width
        out_height = int(round((max_area * aspect_ratio) ** 0.5))
        out_width = int(round(max_area / out_height))

        os.remove(output_path)
        cleanup()

        result = {
            "video": video_base64,
            "format": "mp4",
            "width": out_width,
            "height": out_height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "input_size": [input_width, input_height]
        }

        if extended_prompt:
            result["prompt_extended"] = extended_prompt

        return result

    except torch.cuda.OutOfMemoryError as e:
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("Initializing WAN 2.2 I2V Handler...")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })

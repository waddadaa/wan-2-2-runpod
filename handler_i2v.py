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


def save_temp_file(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


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


def get_image_dimensions(image_bytes: bytes) -> tuple:
    img = Image.open(BytesIO(image_bytes))
    return img.size


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
    image_path = None

    try:
        # =====================================================================
        # REQUIRED: IMAGE
        # =====================================================================
        image_data = job_input.get("image", "")
        if not image_data:
            return {"error": "image is required (base64 encoded)"}

        image_bytes = decode_base64_data(image_data)
        input_width, input_height = get_image_dimensions(image_bytes)
        image_path = save_temp_file(image_bytes, ".png")

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, low resolution, "
            "static, no motion, frozen, ugly, bad anatomy, deformed"
        )

        size = job_input.get("size", "1280x720")
        width, height = parse_resolution(size)

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
        generator = torch.Generator(device="cuda").manual_seed(seed)

        logger.info(f"I2V: {width}x{height}, {num_frames}f, input={input_width}x{input_height}")
        logger.info(f"Sampling: steps={sample_steps}, cfg={sample_guide_scale}, shift={sample_shift}")

        pipeline = load_model()

        # Prompt extension for I2V
        extended_prompt = None
        if use_prompt_extend:
            try:
                from wan.utils.prompt_extend import extend_prompt_vl
                extended_prompt = extend_prompt_vl(
                    image_path,
                    prompt,
                    model=prompt_extend_model,
                    method=prompt_extend_method,
                    target_lang=prompt_extend_target_lang
                )
                logger.info(f"Extended: {extended_prompt[:100]}...")
            except Exception as e:
                logger.warning(f"Prompt extension failed: {e}")

        final_prompt = extended_prompt if extended_prompt else prompt

        with torch.inference_mode():
            output = pipeline(
                image=image_path,
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=sample_guide_scale,
                num_inference_steps=sample_steps,
                generator=generator,
                flow_shift=sample_shift
            )

        output_path = tempfile.mktemp(suffix=".mp4")
        output.save(output_path, fps=fps)
        video_base64 = encode_video_base64(output_path)

        os.remove(output_path)
        os.remove(image_path)
        image_path = None
        cleanup()

        result = {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "input_size": [input_width, input_height]
        }

        if extended_prompt:
            result["prompt_extended"] = extended_prompt

        return result

    except torch.cuda.OutOfMemoryError as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
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

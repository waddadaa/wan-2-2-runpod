"""
WAN 2.2 Speech-to-Video (S2V) Handler for RunPod Serverless
Model: Wan2.2-S2V-14B
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

import runpod

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix safetensors "device cuda:0 is invalid" error
# The safetensors Rust backend has a bug with CUDA device strings
# Workaround: Load checkpoint to CPU RAM first, then model moves tensors to GPU
# This is safe - only checkpoint I/O uses CPU, model runs on GPU
import safetensors.torch
_original_load_file = safetensors.torch.load_file

def _patched_load_file(filename, device="cpu"):
    # Force CPU load to avoid safetensors Rust CUDA bug
    # The model's .to(device) will move tensors to GPU after loading
    result = _original_load_file(filename, device="cpu")
    # If originally requested GPU, move tensors to GPU now
    if device != "cpu" and torch.cuda.is_available():
        target = torch.device("cuda:0")
        result = {k: v.to(target) for k, v in result.items()}
    return result

safetensors.torch.load_file = _patched_load_file
logger.info("Patched safetensors to load via CPU (workaround for Rust CUDA bug)")

# Add WAN 2.2 to path (baked into Docker image)
sys.path.insert(0, "/app/Wan2.2")

MODEL = None

# Model configuration
MODEL_NAME = "Wan2.2-S2V-14B"
HF_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
WAN_CONFIG_KEY = "s2v-14B"


def ensure_model_downloaded(model_dir: str, model_name: str, hf_repo_id: str) -> str:
    """
    Check if model exists, download from HuggingFace if not.
    Returns the path to the model directory.
    """
    ckpt_dir = os.path.join(model_dir, model_name)

    # Check if model already exists (look for key files)
    if os.path.exists(ckpt_dir):
        # Check for typical model files
        has_files = any(
            os.path.exists(os.path.join(ckpt_dir, f))
            for f in ["config.json", "model_index.json", "diffusion_pytorch_model.safetensors"]
        )
        if has_files or len(os.listdir(ckpt_dir)) > 0:
            logger.info(f"Model found at {ckpt_dir}")
            return ckpt_dir

    # Model not found, download from HuggingFace
    logger.info("=" * 60)
    logger.info(f"Model not found at {ckpt_dir}")
    logger.info(f"Downloading {model_name} from HuggingFace...")
    logger.info(f"Repo: {hf_repo_id}")
    logger.info("This may take a while (~43GB download)")
    logger.info("=" * 60)

    try:
        from huggingface_hub import snapshot_download

        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Download the model
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
    """Load the S2V model, downloading from HuggingFace if needed."""
    global MODEL
    if MODEL is not None:
        return MODEL

    # Initialize CUDA before loading model
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        # Warm up CUDA
        _ = torch.zeros(1).cuda()
        logger.info(f"CUDA initialized: {torch.cuda.get_device_name(0)}")

    model_dir = os.environ.get("MODEL_DIR", "/runpod-volume/models")

    # Ensure model is downloaded
    ckpt_dir = ensure_model_downloaded(model_dir, MODEL_NAME, HF_REPO_ID)

    logger.info("=" * 60)
    logger.info("Loading WAN 2.2 S2V model...")
    logger.info(f"Model path: {ckpt_dir}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    try:
        import wan
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS[WAN_CONFIG_KEY]

        MODEL = wan.WanS2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
        )
        logger.info("S2V model loaded successfully!")
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


def get_audio_duration(audio_path: str) -> float:
    try:
        import soundfile as sf
        data, samplerate = sf.read(audio_path)
        return len(data) / samplerate
    except:
        return 5.0


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    WAN 2.2 Speech-to-Video Handler - ALL PARAMETERS

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS (Option A: Pre-recorded Audio)
    ═══════════════════════════════════════════════════════════════════════════

    audio (str):
        Base64 encoded audio file.
        Supports WAV, MP3, FLAC, OGG.
        Speech audio drives lip sync.

    image (str):
        Base64 encoded reference image.
        Clear frontal face recommended.
        Supports PNG, JPG, WEBP.

    ═══════════════════════════════════════════════════════════════════════════
    REQUIRED PARAMETERS (Option B: Text-to-Speech)
    ═══════════════════════════════════════════════════════════════════════════

    enable_tts (bool, default: false):
        Enable text-to-speech synthesis.
        When true, generates audio from text instead of using audio input.

    tts_text (str, required if enable_tts=true):
        Text to synthesize into speech.

    tts_prompt_audio (str, optional):
        Base64 encoded reference voice sample for TTS.
        Model clones this voice.

    tts_prompt_text (str, optional):
        Transcription of tts_prompt_audio.
        Helps model understand reference voice.

    ═══════════════════════════════════════════════════════════════════════════
    GENERATION PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    prompt (str, default: ""):
        Additional text guidance.
        Example: "Professional news anchor in studio"

    negative_prompt (str, default: "..."):
        What to avoid.

    size (str, default: "1280x720"):
        Output resolution.

    num_frames / frame_num (int, default: auto):
        Output frames. Auto-calculated from audio if not set.
        Must be 4n+1 format.

    ═══════════════════════════════════════════════════════════════════════════
    SAMPLING PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    sample_steps / num_inference_steps (int, default: 50):
        Denoising steps.

    sample_guide_scale / guidance_scale (float, default: 5.0):
        CFG scale.

    sample_shift / flow_shift (float, default: 5.0):
        Flow matching shift.

    sample_solver (str, default: "unipc"):
        Solver: "unipc" or "dpm++"

    ═══════════════════════════════════════════════════════════════════════════
    S2V-SPECIFIC PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    num_clip (int, default: 1):
        Number of video clips to generate.
        For long audio, splits into multiple clips.

    infer_frames (int, default: auto):
        Frames per clip inference.

    start_from_ref (bool, default: true):
        Use reference image as starting frame.

    ═══════════════════════════════════════════════════════════════════════════
    OUTPUT PARAMETERS
    ═══════════════════════════════════════════════════════════════════════════

    seed / base_seed (int, default: random):
        Random seed.

    fps (int, default: 24):
        Output FPS. 24 recommended for lip sync.

    ═══════════════════════════════════════════════════════════════════════════
    RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════════════════

    {
        "video": "<base64_mp4>",
        "format": "mp4",
        "width": 1280,
        "height": 720,
        "num_frames": 120,
        "fps": 24,
        "seed": 12345,
        "audio_duration": 5.0,
        "tts_used": false
    }
    """
    job_input = job.get("input", {})
    audio_path = None
    image_path = None
    tts_prompt_audio_path = None

    try:
        # =====================================================================
        # REQUIRED: IMAGE
        # =====================================================================
        image_data = job_input.get("image", "")
        if not image_data:
            return {"error": "image is required (base64 encoded reference image)"}

        image_bytes = decode_base64_data(image_data)
        image_path = save_temp_file(image_bytes, ".png")

        # =====================================================================
        # AUDIO: Either direct audio or TTS
        # =====================================================================
        enable_tts = job_input.get("enable_tts", False)
        tts_used = False

        if enable_tts:
            # TTS Mode
            tts_text = job_input.get("tts_text", "")
            if not tts_text:
                return {"error": "tts_text is required when enable_tts=true"}

            tts_prompt_audio_data = job_input.get("tts_prompt_audio", "")
            tts_prompt_text = job_input.get("tts_prompt_text", "")

            if tts_prompt_audio_data:
                tts_prompt_audio_bytes = decode_base64_data(tts_prompt_audio_data)
                tts_prompt_audio_path = save_temp_file(tts_prompt_audio_bytes, ".wav")

            # Generate audio via TTS
            try:
                from wan.utils.tts import synthesize_speech
                audio_path = tempfile.mktemp(suffix=".wav")
                synthesize_speech(
                    text=tts_text,
                    output_path=audio_path,
                    prompt_audio=tts_prompt_audio_path,
                    prompt_text=tts_prompt_text
                )
                tts_used = True
                logger.info(f"TTS generated: {tts_text[:50]}...")
            except Exception as e:
                return {"error": f"TTS synthesis failed: {str(e)}"}
        else:
            # Direct audio mode
            audio_data = job_input.get("audio", "")
            if not audio_data:
                return {"error": "audio is required (base64 encoded) or enable_tts=true"}

            audio_bytes = decode_base64_data(audio_data)
            if audio_bytes[:4] == b'RIFF':
                audio_suffix = ".wav"
            elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                audio_suffix = ".mp3"
            else:
                audio_suffix = ".wav"
            audio_path = save_temp_file(audio_bytes, audio_suffix)

        audio_duration = get_audio_duration(audio_path)
        logger.info(f"Audio duration: {audio_duration:.2f}s")

        # =====================================================================
        # GENERATION PARAMETERS
        # =====================================================================
        prompt = job_input.get("prompt", "").strip()
        negative_prompt = job_input.get(
            "negative_prompt",
            "poor quality, blurred, distorted, watermark, bad lip sync, "
            "unnatural movement, static, frozen face, ugly"
        )

        size = job_input.get("size", "1280x720")
        width, height = parse_resolution(size)

        fps = max(8, min(60, int(job_input.get("fps", 24))))

        # Auto-calculate frames from audio
        num_frames = job_input.get("num_frames", job_input.get("frame_num", None))
        if num_frames is None:
            num_frames = int(audio_duration * fps) + 1
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
        # S2V-SPECIFIC PARAMETERS
        # =====================================================================
        num_clip = int(job_input.get("num_clip", 1))
        infer_frames = job_input.get("infer_frames", None)
        start_from_ref = job_input.get("start_from_ref", True)

        # =====================================================================
        # OUTPUT PARAMETERS
        # =====================================================================
        seed = job_input.get("seed", job_input.get("base_seed", None))
        if seed is None or seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        seed = int(seed)

        # =====================================================================
        # GENERATE
        # =====================================================================
        logger.info(f"S2V: {width}x{height}, audio={audio_duration:.2f}s")
        logger.info(f"num_clip={num_clip}, start_from_ref={start_from_ref}, seed={seed}")

        wan_s2v = load_model()

        # Calculate max_area from resolution
        max_area = width * height

        # TTS parameters - required positional args even when enable_tts=False
        # If TTS mode was used, audio was already generated above
        tts_prompt_audio_for_gen = None
        tts_prompt_text_for_gen = None
        tts_text_for_gen = None

        # Build generate kwargs
        generate_kwargs = {
            "input_prompt": prompt,
            "ref_image_path": image_path,
            "audio_path": audio_path,
            "enable_tts": False,  # TTS already handled above if needed
            "tts_prompt_audio": tts_prompt_audio_for_gen,
            "tts_prompt_text": tts_prompt_text_for_gen,
            "tts_text": tts_text_for_gen,
            "num_repeat": num_clip,
            "max_area": max_area,
            "shift": sample_shift,
            "sample_solver": sample_solver,
            "sampling_steps": sample_steps,
            "guide_scale": sample_guide_scale,
            "seed": seed,
            "offload_model": True,
            "init_first_frame": start_from_ref,
        }

        if infer_frames is not None:
            generate_kwargs["infer_frames"] = int(infer_frames)

        with torch.inference_mode():
            video_tensor = wan_s2v.generate(**generate_kwargs)

        # Save video
        from wan.utils.utils import save_video
        output_path = tempfile.mktemp(suffix=".mp4")
        save_video(video_tensor, output_path, fps=fps, audio_path=audio_path)
        video_base64 = encode_video_base64(output_path)

        # Cleanup
        os.remove(output_path)
        os.remove(audio_path)
        audio_path = None
        os.remove(image_path)
        image_path = None
        if tts_prompt_audio_path:
            os.remove(tts_prompt_audio_path)
            tts_prompt_audio_path = None
        cleanup()

        return {
            "video": video_base64,
            "format": "mp4",
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "audio_duration": round(audio_duration, 2),
            "tts_used": tts_used
        }

    except torch.cuda.OutOfMemoryError as e:
        for p in [audio_path, image_path, tts_prompt_audio_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        return {"error": "Out of GPU memory", "details": str(e)}
    except Exception as e:
        for p in [audio_path, image_path, tts_prompt_audio_path]:
            if p and os.path.exists(p):
                os.remove(p)
        cleanup()
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


def concurrency_modifier(current_concurrency: int) -> int:
    return 1


if __name__ == "__main__":
    logger.info("Initializing WAN 2.2 S2V Handler...")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Pre-load failed: {e}")

    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": concurrency_modifier,
        "return_aggregate_stream": True
    })

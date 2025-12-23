# WAN 2.2 RunPod Serverless

Deploy WAN 2.2 video generation models on RunPod Serverless with network volume support.

**License**: Apache 2.0 (Commercial use allowed)

## Models

| Model | Handler | Dockerfile | GPU | Size |
|-------|---------|------------|-----|------|
| **Wan2.2-T2V-A14B** | `handler_t2v_14b.py` | `Dockerfile.t2v-14b` | A100 80GB | ~50GB |
| **Wan2.2-TI2V-5B** | `handler_ti2v_5b.py` | `Dockerfile.ti2v-5b` | RTX 4090 24GB | ~33GB |
| **Wan2.2-I2V-A14B** | `handler_i2v.py` | `Dockerfile.i2v` | A100 80GB | ~50GB |
| **Wan2.2-S2V-14B** | `handler_s2v.py` | `Dockerfile.s2v` | A100 80GB | ~43GB |
| **Wan2.2-Animate-14B** | `handler_v2v.py` | `Dockerfile.v2v` | A100 80GB | ~50GB |

## Prerequisites

### Network Volume Structure

Create a RunPod network volume (200GB+ recommended) mounted at `/runpod-volume`:

```
/runpod-volume/
├── models/
│   ├── Wan2.2-T2V-A14B/           # ~50GB (auto-downloaded)
│   ├── Wan2.2-TI2V-5B/            # ~33GB (auto-downloaded)
│   ├── Wan2.2-I2V-A14B/           # ~50GB (auto-downloaded)
│   ├── Wan2.2-S2V-14B/            # ~43GB (auto-downloaded)
│   ├── Wan2.2-Animate-14B/        # ~50GB (auto-downloaded)
│   └── preprocess/                # For Animate model
│       ├── det/yolov10m.onnx          # ~25MB (auto-downloaded)
│       └── pose2d/vitpose_h_wholebody.onnx  # ~1GB (auto-downloaded)
└── huggingface-cache/             # HF cache directory
```

### Download Models (Optional)

Models are **auto-downloaded** on first request. To pre-download manually:

```bash
# Download models using huggingface-cli
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir /runpod-volume/models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /runpod-volume/models/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /runpod-volume/models/Wan2.2-I2V-A14B
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /runpod-volume/models/Wan2.2-S2V-14B
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir /runpod-volume/models/Wan2.2-Animate-14B
```

For Animate preprocessing models (also auto-downloaded):
```bash
# YOLOv10m detector
wget -P /runpod-volume/models/preprocess/det/ \
  https://huggingface.co/onnx-community/yolov10m/resolve/main/onnx/model.onnx \
  -O /runpod-volume/models/preprocess/det/yolov10m.onnx

# ViTPose wholebody
wget -P /runpod-volume/models/preprocess/pose2d/ \
  https://huggingface.co/wanghaofan/Sonic/resolve/main/vitpose-h-wholebody.onnx \
  -O /runpod-volume/models/preprocess/pose2d/vitpose_h_wholebody.onnx
```

## Build & Deploy

### Build Docker Images

```bash
# T2V 14B (Text-to-Video) - Requires A100 80GB
docker build -f Dockerfile.t2v-14b -t YOUR_USERNAME/wan-t2v-14b:latest .
docker push YOUR_USERNAME/wan-t2v-14b:latest

# TI2V 5B (Text+Image-to-Video) - Works on RTX 4090
docker build -f Dockerfile.ti2v-5b -t YOUR_USERNAME/wan-ti2v-5b:latest .
docker push YOUR_USERNAME/wan-ti2v-5b:latest

# I2V (Image-to-Video) - Requires A100 80GB
docker build -f Dockerfile.i2v -t YOUR_USERNAME/wan-i2v:latest .
docker push YOUR_USERNAME/wan-i2v:latest

# S2V (Speech-to-Video) - Requires A100 80GB
docker build -f Dockerfile.s2v -t YOUR_USERNAME/wan-s2v:latest .
docker push YOUR_USERNAME/wan-s2v:latest

# V2V/Animate (Video-to-Video) - Requires A100 80GB
docker build -f Dockerfile.v2v -t YOUR_USERNAME/wan-v2v:latest .
docker push YOUR_USERNAME/wan-v2v:latest
```

### RunPod Endpoint Settings

| Setting | Value |
|---------|-------|
| Container Image | `YOUR_USERNAME/wan-XXX:latest` |
| Network Volume | Mount at `/runpod-volume` |
| GPU | A100 80GB (14B models) / RTX 4090 (5B model) |
| Active Workers | 0 |
| Max Workers | 1-3 |
| Idle Timeout | 60s |
| Execution Timeout | 600s |

> **Note:** Models are auto-downloaded on first request if not present on the network volume.

---

## API Reference

### T2V-14B (Text-to-Video)

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description of the video |

**Optional:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | auto | What to avoid |
| `size` | string | "1280x720" | Resolution (WIDTHxHEIGHT) |
| `num_frames` | int | 81 | Frame count (must be 4n+1) |
| `sample_steps` | int | 50 | Denoising steps |
| `guidance_scale` | float | 5.0 | CFG scale |
| `seed` | int | random | Random seed (-1 for random) |
| `fps` | int | 16 | Output FPS |

**Example:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing with a ball in a sunny garden",
      "size": "1280x720",
      "num_frames": 81,
      "sample_steps": 50,
      "guidance_scale": 5.0,
      "fps": 16
    }
  }'
```

---

### TI2V-5B (Text+Image-to-Video)

Dual-mode model supporting both T2V (text-only) and I2V (text+image) generation.
Smallest model - works on RTX 4090 24GB.

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description |

**Optional - Mode Selection:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string | Base64 image. If provided: I2V mode. If omitted: T2V mode |

**Optional - Generation:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | from config | What to avoid |
| `size` | string | "1280x704" | Resolution (720P) |
| `frame_num` | int | 121 | Frame count (4n+1) |

**Optional - Sampling:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_steps` | int | T2V=50, I2V=40 | Denoising steps (mode-dependent) |
| `guide_scale` | float | 5.0 | CFG scale |
| `shift` | float | 5.0 | Flow shift (use 3.0 for 480p) |
| `sample_solver` | string | "unipc" | Solver: "unipc" or "dpm++" |

**Optional - Output:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | random | Random seed |
| `fps` | int | 24 | Output FPS |
| `offload_model` | bool | true | Offload to CPU (set false for A100) |

**Example (T2V mode - text only):**
```python
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "prompt": "A cat playing with a ball in a sunny garden",
            "size": "1280x704",
            "frame_num": 121,
            "fps": 24
        }
    }
)
```

**Example (I2V mode - with image):**
```python
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "prompt": "The woman turns and smiles",
            "image": "<base64_encoded_image>",
            "size": "1280x704",
            "frame_num": 121
        }
    }
)
```

---

### I2V (Image-to-Video)

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string | Base64 encoded image |

**Optional:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Motion description |
| `size` | string | "1280x720" | Resolution |
| `num_frames` | int | 81 | Frame count |
| `sample_steps` | int | 50 | Denoising steps |
| `guidance_scale` | float | 5.0 | CFG scale |
| `seed` | int | random | Random seed |
| `fps` | int | 16 | Output FPS |

---

### S2V (Speech-to-Video)

Generates talking head videos with lip-sync from audio or text-to-speech.

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string | Base64 face image (frontal recommended) |

**Audio Input (Option A: Pre-recorded):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | string | Base64 audio file (WAV/MP3/FLAC) |

**Audio Input (Option B: Text-to-Speech):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `enable_tts` | bool | Set to `true` |
| `tts_text` | string | Text to synthesize |
| `tts_prompt_audio` | string | (Optional) Base64 voice sample for cloning |
| `tts_prompt_text` | string | (Optional) Transcription of voice sample |

> **Note:** TTS uses CosyVoice2, downloaded on first use (~1GB).

**Optional - Generation:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Style guidance |
| `negative_prompt` | string | from config | What to avoid |
| `max_area` | int | 921600 | Max pixel area (1280x720) |

**Optional - Sampling:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sampling_steps` | int | 40 | Denoising steps |
| `guide_scale` | float | 4.5 | CFG scale |
| `shift` | float | 3.0 | Flow matching shift |
| `sample_solver` | string | "unipc" | Solver: "unipc" or "dpm++" |

**Optional - S2V Specific:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_repeat` | int | auto | Clips for long audio |
| `infer_frames` | int | 80 | Frames per clip (must be 4n) |
| `init_first_frame` | bool | false | Use ref as exact first frame |
| `pose_video` | string | null | Base64 pose video for pose-driven generation |

**Optional - Output:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | random | Random seed |
| `fps` | int | 16 | Output FPS |
| `offload_model` | bool | true | Offload to CPU after generation |

**Example (Audio mode):**
```python
import base64
import requests

with open("face.png", "rb") as f:
    image = base64.b64encode(f.read()).decode()
with open("speech.wav", "rb") as f:
    audio = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "image": image,
            "audio": audio,
            "sampling_steps": 40,
            "guide_scale": 4.5
        }
    }
)
```

**Example (TTS mode):**
```python
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "image": image,
            "enable_tts": True,
            "tts_text": "Hello, this is a test of the speech to video system.",
            "sampling_steps": 40
        }
    }
)
```

---

### Animate (Motion Transfer)

Transfers motion from a driving video to a reference image character.

```
INPUT:
├── reference_image.png   → Person A (the appearance you want)
└── driving_video.mp4     → Person B moving (the motion you want)

OUTPUT:
└── output.mp4            → Person A doing Person B's movements
```

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_image` | string | Base64 encoded image (character appearance) |
| `driving_video` | string | Base64 encoded video (motion source) |

**Optional - Preprocessing:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resolution` | list | [1280, 720] | Output resolution [width, height] |
| `fps` | int | 30 | Frames per second |
| `retarget_flag` | bool | false | Enable pose retargeting for different body proportions |

**Optional - Generation:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Text description (minimal effect) |
| `negative_prompt` | string | "" | What to avoid |
| `clip_len` | int | 77 | Frames per clip (must be 4n+1) |
| `refert_num` | int | 1 | Temporal guidance frames (1 or 5) |
| `sampling_steps` | int | 20 | Denoising steps |
| `guide_scale` | float | 1.0 | CFG scale |
| `shift` | float | 5.0 | Flow matching shift |
| `sample_solver` | string | "dpm++" | Solver: "dpm++" or "unipc" |
| `seed` | int | random | Random seed (-1 for random) |
| `offload_model` | bool | true | Offload to CPU to save VRAM |

> **Note:** `use_relighting_lora` is always enabled for better lighting quality.

**Example:**
```python
import base64
import requests

# Encode files
with open("person.png", "rb") as f:
    ref_image = base64.b64encode(f.read()).decode()
with open("dance.mp4", "rb") as f:
    drive_video = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "reference_image": ref_image,
            "driving_video": drive_video,
            "resolution": [1280, 720],
            "fps": 30,
            "retarget_flag": False
        }
    }
)

# Save output
result = response.json()
video_bytes = base64.b64decode(result["output"]["video"])
with open("output.mp4", "wb") as f:
    f.write(video_bytes)
```

**When to use `retarget_flag`:**
| Scenario | retarget_flag |
|----------|---------------|
| Same body proportions | `false` |
| Different body sizes (tall/short) | `true` |
| Different body types | `true` |

---

## Response Format

**T2V / I2V:**
```json
{
  "video": "<base64_encoded_mp4>",
  "format": "mp4",
  "width": 1280,
  "height": 720,
  "num_frames": 81,
  "fps": 16,
  "seed": 12345
}
```

**TI2V-5B:**
```json
{
  "video": "<base64_encoded_mp4>",
  "format": "mp4",
  "width": 1280,
  "height": 704,
  "frame_num": 121,
  "fps": 24,
  "seed": 12345,
  "mode": "T2V" or "I2V"
}
```

**S2V:**
```json
{
  "video": "<base64_encoded_mp4>",
  "format": "mp4",
  "fps": 16,
  "seed": 12345,
  "audio_duration": 5.0,
  "tts_used": false
}
```

**Animate:**
```json
{
  "video": "<base64_encoded_mp4>",
  "format": "mp4",
  "resolution": [1280, 720],
  "fps": 30,
  "seed": 12345
}
```

## File Structure

```
wan-2-2-serverless/
├── Dockerfile.t2v-14b      # T2V 14B
├── Dockerfile.ti2v-5b      # TI2V 5B
├── Dockerfile.i2v          # I2V 14B
├── Dockerfile.s2v          # S2V 14B
├── Dockerfile.v2v          # Animate 14B
├── handler_t2v_14b.py
├── handler_ti2v_5b.py
├── handler_i2v.py
├── handler_s2v.py
├── handler_v2v.py
├── requirements.txt
├── setup_volume.sh
└── README.md
```

## Links

- [WAN 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Hugging Face Models](https://huggingface.co/Wan-AI)
- [RunPod Documentation](https://docs.runpod.io/)

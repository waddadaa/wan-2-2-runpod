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

Create a RunPod network volume (200GB+ recommended) mounted at `/workspace`:

```
/workspace/
├── Wan2.2/                    # Cloned repository
├── models/
│   ├── Wan2.2-T2V-A14B/
│   ├── Wan2.2-TI2V-5B/
│   ├── Wan2.2-I2V-A14B/
│   ├── Wan2.2-S2V-14B/
│   └── Wan2.2-Animate-14B/
└── env/                       # Python environment (optional)
```

### Download Models

```bash
# Clone WAN 2.2 repository
git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /workspace/Wan2.2

# Download models using huggingface-cli
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir /workspace/models/Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /workspace/models/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /workspace/models/Wan2.2-I2V-A14B
huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir /workspace/models/Wan2.2-S2V-14B
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir /workspace/models/Wan2.2-Animate-14B
```

Or use the setup script:
```bash
./setup_volume.sh
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
| Network Volume | Mount at `/workspace` |
| GPU | A100 80GB (14B models) / RTX 4090 (5B model) |
| Active Workers | 0 |
| Max Workers | 1-3 |
| Idle Timeout | 60s |
| Execution Timeout | 600s |

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

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | string | Text description |
| `image` | string | Base64 encoded input image |

**Optional:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | string | "832x480" | Resolution |
| `num_frames` | int | 81 | Frame count |
| `sample_steps` | int | 50 | Denoising steps |
| `guidance_scale` | float | 5.0 | CFG scale |
| `seed` | int | random | Random seed |
| `fps` | int | 16 | Output FPS |

**Example:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "The woman turns and smiles",
      "image": "<base64_encoded_image>",
      "size": "832x480",
      "num_frames": 81
    }
  }'
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

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string | Base64 face image (frontal recommended) |
| `audio` | string | Base64 audio file (WAV/MP3) |

**Or for TTS mode:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | string | Base64 face image |
| `enable_tts` | bool | Set to `true` |
| `tts_text` | string | Text to synthesize |

**Optional:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | "" | Additional guidance |
| `size` | string | "1280x720" | Resolution |
| `num_frames` | int | auto | Auto from audio duration |
| `sample_steps` | int | 50 | Denoising steps |
| `guidance_scale` | float | 5.0 | CFG scale |
| `num_clip` | int | 1 | Clips for long audio |
| `start_from_ref` | bool | true | Use ref as first frame |
| `seed` | int | random | Random seed |
| `fps` | int | 24 | Output FPS (24 for lip sync) |

**Example:**
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "<base64_face_image>",
      "audio": "<base64_audio>",
      "prompt": "Professional speaker",
      "size": "1280x720",
      "fps": 24
    }
  }'
```

---

### V2V/Animate (Video-to-Video)

**Required:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `video` | string | Base64 source video |

**Optional:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_image` | string | null | Base64 character image |
| `prompt` | string | "" | Output description |
| `size` | string | "1280x720" | Resolution |
| `num_frames` | int | 81 | Frame count |
| `sample_steps` | int | 50 | Denoising steps |
| `guidance_scale` | float | 5.0 | CFG scale |
| `refert_num` | int | 1 | Temporal guidance (1 or 5) |
| `replace_flag` | bool | false | Character replacement mode |
| `use_relighting_lora` | bool | false | Relighting LoRA |
| `seed` | int | random | Random seed |
| `fps` | int | 16 | Output FPS |

---

## Response Format

```json
{
  "video": "<base64_encoded_mp4>",
  "format": "mp4",
  "width": 1280,
  "height": 720,
  "num_frames": 81,
  "fps": 16,
  "seed": 12345,
  "model": "Wan2.2-T2V-A14B"
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

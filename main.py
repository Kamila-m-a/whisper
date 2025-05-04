from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import numpy as np
import os
import warnings
import subprocess

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
np.zeros(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("tiny", device="cpu")
model.eval()
torch.set_num_threads(1)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    input_path = "input_audio"
    output_path = "converted_audio.wav"
    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Convert to 16kHz mono WAV using ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", output_path],
            check=True
        )

        with torch.inference_mode():
            result = model.transcribe(output_path, fp16=False)

        return {"text": result["text"]}

    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Failed to convert audio.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "tiny",
        "info": "POST /transcribe to get transcript"
    }

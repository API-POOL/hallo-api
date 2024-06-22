import sys
import fastapi
from pydantic import BaseModel
from typing import Optional
import uvicorn
from scripts.inference import inference_process
from utils import b64_to_audio, b64_to_image, video_to_b64

app = fastapi.FastAPI()
image_path = ".cache/image.jpg"
audio_path = ".cache/audio.wav"
video_path = ".cache/video.mp4"
     
class Body(BaseModel):
    image: str
    audio: str
    
    pose_weight: float = 1.0
    face_weight: float = 1.0
    lip_weight: float = 1.0
    face_expand_ratio: float = 1.2
    setting_steps: int = 40
    setting_cfg: float = 3.5
    settings_seed: int = 42
    settings_fps: int = 25
    settings_motion_pose_scale: float = 1.1
    settings_motion_face_scale: float = 1.1
    settings_motion_lip_scale: float = 1.1
    settings_n_motion_frames: int = 2
    settings_n_sample_frames: int = 16
   
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/")
def generate(body: Body):
    b64_to_audio(body.audio, audio_path)
    b64_to_image(body.image, image_path)
    
    body.soruce_image = image_path
    body.driving_audio = audio_path
    
    body.output = video_path
    body.config = "configs/inference/default.yaml"
    body.checkpoint = None
    
    inference_process(
        body,
        body.setting_steps,
        body.setting_cfg,
        body.settings_seed,
        body.settings_fps,
        body.settings_motion_pose_scale,
        body.settings_motion_face_scale,
        body.settings_motion_lip_scale,
        body.settings_n_motion_frames,
        body.settings_n_sample_frames
    )
    
    b64_video = video_to_b64(video_path)
    
    return {"video": b64_video}
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
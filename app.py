import sys
import fastapi
from pydantic import BaseModel
from typing import Optional
import uvicorn
from scripts.inference import inference_process
from utils import b64_to_audio, b64_to_image, video_to_b64

app = fastapi.FastAPI()
image_path = "/app/input-image.jpg"
audio_path = "/app/input-audio.wav"
video_path = "/app/output-video.mp4"
     
class Body(BaseModel):
    image: str
    audio: str
    
    source_image: Optional[str] = None
    driving_audio: Optional[str] = None
    output: Optional[str] = None
    config: Optional[str] = None
    checkpoint: Optional[str] = None
    
    pose_weight: Optional[float] = 1.0
    face_weight: Optional[float] = 1.0
    lip_weight: Optional[float] = 1.0
    face_expand_ratio: Optional[float] = 1.2
    setting_steps: Optional[int] = 40
    setting_cfg: Optional[float] = 3.5
    settings_seed: Optional[int] = 42
    settings_fps: Optional[int] = 25
    settings_motion_pose_scale: Optional[float] = 1.1
    settings_motion_face_scale: Optional[float] = 1.1
    settings_motion_lip_scale: Optional[float] = 1.1
    settings_n_motion_frames: Optional[int] = 2
    settings_n_sample_frames: Optional[int] = 16
   
@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/")
def generate(body: Body):
    b64_to_audio(body.audio, audio_path)
    b64_to_image(body.image, image_path)
    
    setattr(body, "source_image", image_path)
    setattr(body, "driving_audio", audio_path)
    
    setattr(body, "output", video_path)
    setattr(body, "config", "configs/inference/default.yaml")
    setattr(body, "checkpoint", None)
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
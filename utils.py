import io
from PIL import Image
import base64

def video_to_b64(video_path):
    with open(video_path, "rb") as video_file:
        video_content = video_file.read()
    return base64.b64encode(video_content).decode("utf-8")

def b64_to_image(b64_str, image_path):
    image_data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    
def b64_to_audio(b64_str, audio_path):
    audio_data = base64.b64decode(b64_str)
    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_data)
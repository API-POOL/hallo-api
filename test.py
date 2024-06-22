import utils as u
import base64

with open("./a.wav", "rb") as f:
    audio = f.read()
    base64_audio = base64.b64encode(audio).decode("utf-8")

with open("./te/a.txt", "w") as f:
    f.write(base64_audio)

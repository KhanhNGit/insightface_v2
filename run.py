from insightface.face_analysis import FaceAnalysis
from insightface.image import get_image
import time

app = FaceAnalysis()
app.prepare()

start = time.time()
img = get_image('2')
faces = app.get(img)
end = time.time()
print(end-start)
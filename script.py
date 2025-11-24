import torch
from torchvision import transforms
from PIL import Image
import torchvision
from torch import nn
import cv2 as cv



face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

class FERmodel(nn.Module):
  def __init__(self):
    super(FERmodel,self).__init__()
    self.conv_layers = nn.Sequential(
         nn.Conv2d(1,32,kernel_size=3, padding=1),
         nn.BatchNorm2d(32),
         nn.ReLU(),
         nn.MaxPool2d(2,2),

         nn.Conv2d(32,64,kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(),


         nn.Conv2d(64,64,kernel_size=3, padding=1),
         nn.BatchNorm2d(64),
         nn.ReLU(),
         #nn.MaxPool2d(2,2),

         nn.Conv2d(64,128,kernel_size=3, padding=1),
         nn.BatchNorm2d(128),
         nn.ReLU(),
         nn.MaxPool2d(2,2),

         nn.Conv2d(128,256,kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(),
         #nn.MaxPool2d(2,2),
         nn.Conv2d(256,256,kernel_size=3, padding=1),
         nn.BatchNorm2d(256),
         nn.ReLU(),

         nn.Conv2d(256,512,kernel_size=3, padding=1),
         nn.BatchNorm2d(512),
         nn.ReLU(),
         nn.MaxPool2d(2,2)
     )
    self.fc_layers = nn.Sequential(
        nn.Linear(512*6*6,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512,8),
    )
  def forward(self,x):
    x = self.conv_layers(x)
    x = x.view(x.size(0),-1)
    x = self.fc_layers(x)
    return x

model = FERmodel()
model.load_state_dict(torch.load("FERmodel_NewARCH_epoch3.pt"))
model.eval()

label_map = [
        'surprise',
        'anger',
        'disgust',
        'fear',
        'sad',
        'contempt',
        'neutral',
        'happy'
]

cap = cv.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces)> 0 :
        for x,y,w,h in faces:
            cv.rectangle(frame, (x, y), (x + w+20, y + h+20), (255, 255, 0), 2)
            roi = frame[y:y+h, x:x+w]

            rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            image_tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                 output = model(image_tensor)
                 pred = int(torch.argmax(output, dim=1).item())
                 emotion = label_map[pred]
                 cv.putText(frame,emotion,(x + w // 2, y + h + 50), cv.FONT_HERSHEY_TRIPLEX,1,color=(255,255,0),thickness=2)
            cv.imshow("frame", frame)

    if cv.waitKey(1) & 0xff == ord("q"):
        break
cap.release()
cv.destroyAllWindows()







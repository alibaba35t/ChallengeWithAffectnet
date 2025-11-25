# ChallengeWithAffectnet – From Scratch Emotion Recognition
[![Status: Training](https://img.shields.io/badge/Status-Active-green)](https://github.com/alibaba35t/ChallengeWithAffectnet)
[![PyTorch](https://img.shields.io/badge/made_with-pytorch-EE4C2C?logo=pytorch&style=flat&logoColor=white)](#)
![From Scratch](https://img.shields.io/badge/Training-From%20Scratch-critical)

This is my first FER(face emotion recognition ) model , I am building it for my future projects.


## Introduction
Heyy, I am Emre. I am a Data Science student in Venice Ca' Foscari University . I am waiting for my study visa and I hate waiting (That's why I don't sit idle while waiting, I dive right into the industry). I explored my interests about machine/deep learning and since the summer, I have been trying to improve myself by working on **Pytorch**, **computer vision** and **data science** along with my 2 years of software experience.

## Main Goals 
- Build my own image processing model without using pre-trained models( both learning data science. (**layers**, **neural networks**) and using my next projects (my own pipeline))
- Make my model's accuracy score above %75. (I will compare with pretrained model)
- Use this model with OpenCV and webcam. (live emotion recognition)

  <br>

## Testing Script Result
- I built a basic script to test my trained model. (https://github.com/alibaba35t/ChallengeWithAffectnet/blob/main/script.py)
- Honestly, it has above-average performance. However, sometimes it can behave eratically when choosing spesific emotions (like contempt and happy).
- In addition, **environmental features** effects model's deciding process.(for instance; hair, facial hair, light, sunglasses)
- In conclusion, accuracy not sufficient for me and I have to make some differences to improve my model.

<br>

## What's next?
- New scheduler (I used **ReduceLROnPlateau** but I'll change it with ***CosineAnnealingLR***, because I have to try it new things)
- New optimizer (In previous examples, I work with Adamax and I decided to change it with AdamW)
- One more csv cleaning (but now, according to relFCs column)
  
<br>

## Today's Reports
**19/11/2025(accuracy: %36 -> %58)**
- I considered that my script(https://github.com/alibaba35t/LabelProblemSolution) made a few mistake when choose some files subset. I will handle it.
- I am using Google Colab Notebook and some runtime issues raised yesterday. Finally, I saved my model after every epoch.
- I am new in image processing and it's concepts in machine learning. Therefore, I may not have used some method correctly or choosen better options.
- Some keys and **layers** really detailed. If ı know their mathematical background, i feel comfortable

**20/11/2025(accuracy: %28 -> %64)**
- I observed an accuracy peak(Initial:%28, Last: %64). Now, my runtime disconnected and I am writing last updates.
- I learned colorJitter(in transform) and label smoothing(in CrossEntropyLoss function). I will compare final accuracies with new implementations. Maybe, I will make changes about conv2d layers.
- I trained my saved model .(which have the lowest loss)

**22-23/11/2025(accuracy: %38 -> %72) (with new model archtitecture)**  
- I changed my model architecture so I started over evaluation process.
- I studied layers and in new model, we have more features and **BatchNorms**.
- I examined some pretrained models like MobileNet and it's architecture to make more accurate model. (It is may way to learn I usually research well)
- New model is heavier than old model. For istance, in old model saved models size usually 3.2 MB but in new model, it's size over 40 MB. Accordingly, new model is slower.

**24/11/2025(accuracy: %52 -> %66) (with scheduler, optimizer and images which have >0.8 relFC)**  
- I changed my scheduler (**ReduceLROnPlateau** to ***CosineAnnealingLR***) and optimizer (**Adamax** to **AdamW**).
- In addition, I trained my model with more clear images which have only 0.8 and upper relFC score in csv. (Total image count: 28.000 -> 15.000) 
- Initially, I am not satisfied with final results. Because, final accuracy score is %66. However, our loss was higher than 1.0 in previous tests. Today, it reduced around 0.66.  
- The most interesting things is (***I WANT TO TAKE ATTENTION TO THIS PART***) I considered final model can decide emotions better with live webcam demo. According to my test, **Loss** is as crucial as **Accuracy**.
- I created a new strategy: **Emotion Based Training**

  <br>

## Compare

|   Features                            |  Old Model                             | New Model                              | 
|:--------------------------------------|:--------------------------------------:|:--------------------------------------:|
| Features Diagram                      | 1 - 32 - 64 - 128 (output)             | 1-32-64-64-128-256-256-512 (output)    |
| Number of Conv2d Layer                | 4                                      | 8                                      |
| BatchNorm                             | ❌                                     | ✅                                     |
| Label Smoothing                       | Added later                            | Default                                |
| Highest Accuracy                      | %64                                    | **%72 (for now)**                      |
| Time (one epoch to another)           | Generally 5 or 6                       | 12-13 (Sometimes 20)                   |
| Scheduler                             | ❌                                     | ReduceLROnPlateu                       |       



### Some pictures 

20/11/2025

<img width="632" height="406" alt="Screenshot from 2025-11-20 21-49-00" src="https://github.com/user-attachments/assets/77b0295f-7048-470b-a338-92ba003ca61a" />


22-23/11/2025

<img width="620" height="299" alt="Screenshot from 2025-11-22 23-10-59" src="https://github.com/user-attachments/assets/18c85c0d-d1e3-45d5-ae54-6db6c43e0011" />
<img width="607" height="270" alt="Screenshot from 2025-11-23 14-28-10" src="https://github.com/user-attachments/assets/ff17d5d5-f879-4e17-b4e7-fef58fe71c59" />

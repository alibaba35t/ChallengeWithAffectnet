# ChallengeWithAffectnet
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

### Some pictures 

20/11/2025


<img width="632" height="406" alt="Screenshot from 2025-11-20 21-49-00" src="https://github.com/user-attachments/assets/77b0295f-7048-470b-a338-92ba003ca61a" />

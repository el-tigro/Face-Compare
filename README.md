# FaceCompare

## About

When applying for a loan, the client is required to upload the following photos:  
- photo: A selfie with the passport held in hand.  
- passport: A photo of the passport's main page.  

This service is designed to detect faces in the photos:  
- big_face: The selfie.  
- small_face: The small photos from the passport.  

Features extracted from these faces:  
- confidence: The quality of the recognized face.  
- share: The proportion of pixels occupied by each face relative to the entire photo.  
- similarity: The cosine distance between the faces.  

The goal of this service is to perform pairwise similarity comparison of two faces in the first photo, specifically comparing the face in the selfie with the face in the passport photo. Additionally, it involves pairwise comparison of the two faces in the first photo with a single face in the second photo.  
The passport photo from the selfie and the standalone passport photo should be the most similar, but it is suspicious if they match 100%.  
If the cosine distance between the embeddings of two faces is less than 0, it is highly likely that they are either different people or the photo is very blurry.


The service takes the Company name and Application ID as input.  
The service outputs a JSON with various variables.   


## Getting started

Add variables to .env file  
```
AWS_ACCESS_KEY=your_access_key_here
AWS_SECRET_KEY=your_secret_key_here

fastapi run  main.py --port 8666
```

 http://localhost:8666/docs#/default/get_face_compare_face_compare_post


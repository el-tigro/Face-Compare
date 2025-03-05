import pandas as pd
import cv2
import numpy as np
from itertools import combinations_with_replacement

# import ssim
import logging
from functools import wraps
import time
from copy import deepcopy
from dotenv import load_dotenv

from load_data import *

logger = logging.getLogger(__name__)
load_dotenv()


@time_it
def mirror_face(face_image):
    """Function for creating a mirror reflection of the face"""
    return cv2.flip(face_image, 1)  # 1 means vertical reflection


@time_it
def get_face_info(faces, img, document_type):
    """Function for extracting embeddings and face information"""
    face_cnt = 0
    face_info = []
    img_height, img_width = img.shape[:2]
    for i, face in enumerate(faces):
        # Face coordinates
        x1, y1, x2, y2 = face.bbox.astype(int)
        # Face dimensions
        width = x2 - x1
        height = y2 - y1
        # Face area in pixels
        size = width * height
        # Proportion of the face relative to the image
        share = (width * height) / (img_width * img_height)
        # Type of face photo (selfie or passport photo)
        if (document_type == 'photo') and (share >= 0.038):
            photo_type = 'big_face'
        if (document_type == 'photo') and (share < 0.038):
            photo_type = 'small_face'
        if (document_type == 'passport') and (share < 0.5):
            photo_type = 'small_face'
        if (document_type == 'passport') and (share >= 0.5):
            photo_type = 'big_face'
        # Face recognition confidence
        confidence = face.det_score
        # Number of faces:
        if confidence > 0.65:
            face_cnt += 1
        # Face embedding
        embedding = face.embedding
        # # Mirror reflection of the face
        # mirrored_face = mirror_face(img[max([y1, 0]):y2, max([x1, 0]):x2])
        # # Embedding of the mirrored face
        # mirrored_embedding = app.get(mirrored_face)[0].embedding if len(app.get(mirrored_face)) > 0 else None

        face_info.append({
            'index': i + 1,
            'bbox': (x1, y1, x2, y2),
            'size': size,
            'share': share,
            'photo_type': photo_type,
            'confidence': confidence,
            'embedding': embedding,
            # 'mirrored_embedding': mirrored_embedding
        })
    return face_info, face_cnt


@time_it
def cosine_similarity(embedding1, embedding2):
    """Function for calculating cosine similarity"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


@time_it
def compare_faces(document_type1, document_type2, faces_info_dict):
    """Face comparison"""
    compare_info = []
    for face1 in faces_info_dict[document_type1]:
        for face2 in faces_info_dict[document_type2]:
            if ((document_type1 != document_type2) | (face1['index'] < face2['index'])):
                # Comparison of original faces
                similarity = cosine_similarity(face1['embedding'], face2['embedding'])

                compare_info.append([
                    f"{document_type1}_{face1['photo_type']}",
                    f"{document_type2}_{face2['photo_type']}",
                    face1['share'],
                    face2['share'],
                    face1['confidence'],
                    face2['confidence'],
                    similarity
                ])

    return pd.DataFrame(
        compare_info,
        columns=[
            'document_photo_type1', 'document_photo_type2',
            'share1', 'share2',
            'confidence1', 'confidence2',
            'similarity'])


@time_it
def extract_face_compare_features(company_name, df, model):
    dict_face_compare_features = {}
    hashed_file_name = {}
    downloaded_image_path = {}
    img_photo_dict = {}
    faces_photo_dict = {}
    faces_info_dict = {}

    start_time = dt.datetime.now()
    logger.info("Start processing photos")

    for document_type in ['photo', 'passport']:
        if (df.name == document_type).sum() > 0:
            hashed_file_name[document_type] = df[df.name == document_type]['hashedFilename'].iloc[-1]
            dict_face_compare_features[f'hashed_file_name_{document_type}'] = hashed_file_name[document_type]

            start_time_locale = dt.datetime.now()
            downloaded_image_path[document_type] = download_file_from_s3(
                company_name=company_name,
                hashed_file_name=hashed_file_name[document_type],
                download_folder=download_folder
            )
            logger.info(
                f"Time used for downloading = {round((dt.datetime.now() - start_time_locale).total_seconds(), 1)} seconds \n"
            )
            # image = Image.open(downloaded_image_path[document_type])
            # display(image)

            # Loading images
            img_photo_dict[document_type] = cv2.imread(downloaded_image_path[document_type])

            # Face detection and embedding extraction
            faces_photo_dict[document_type] = model.get(img_photo_dict[document_type])

            # Getting face information
            faces_info_dict[document_type], face_cnt = get_face_info(faces_photo_dict[document_type],
                                                                     img_photo_dict[document_type], document_type)
            dict_face_compare_features[f'face_cnt_{document_type}'] = face_cnt

            # Deleting the image from local storage
            start_time_locale = dt.datetime.now()
            remove_file_from_s3(downloaded_image_path[document_type])
            logger.info(
                f"Time used for removing from local storage = {round((dt.datetime.now() - start_time_locale).total_seconds(), 1)} seconds \n"
            )

    for document_type1, document_type2 in combinations_with_replacement(list(faces_info_dict), 2):
        df_compare_faces = compare_faces(document_type1, document_type2, faces_info_dict)

        for i in range(len(df_compare_faces)):
            photo_type1 = df_compare_faces.iloc[i]['document_photo_type1']
            photo_type2 = df_compare_faces.iloc[i]['document_photo_type2']

            for similarity_name in [f"similarity_{photo_type1}_{photo_type2}",
                                    f"similarity_{photo_type2}_{photo_type1}"]:
                if similarity_name in [
                    "similarity_photo_big_face_photo_small_face",
                    "similarity_photo_big_face_passport_small_face",
                    "similarity_photo_small_face_passport_small_face"]:
                    if similarity_name not in dict_face_compare_features.keys():
                        dict_face_compare_features[similarity_name] = -999
                    if df_compare_faces.iloc[i]['similarity'] > dict_face_compare_features[similarity_name]:
                        dict_face_compare_features[similarity_name] = df_compare_faces.iloc[i]['similarity']
                        dict_face_compare_features[f"share_{photo_type1}"] = df_compare_faces.iloc[i]['share1']
                        dict_face_compare_features[f"share_{photo_type2}"] = df_compare_faces.iloc[i]['share2']
                        dict_face_compare_features[f"confidence_{photo_type1}"] = df_compare_faces.iloc[i][
                            'confidence1']
                        dict_face_compare_features[f"confidence_{photo_type2}"] = df_compare_faces.iloc[i][
                            'confidence2']

    logger.info(
        f"Total time used for the service = {round((dt.datetime.now() - start_time).total_seconds(), 1)} seconds \n"
    )

    del hashed_file_name, img_photo_dict, faces_photo_dict, faces_info_dict
    return dict_face_compare_features
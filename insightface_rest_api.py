"""
This module is created to get the encodings of the faces present in a batch of frames

Author: Hamza Aziz and Kshitij Parashar
"""

import requests
import cv2
import base64
import msgpack
import ujson
import numpy


INSIGHTFACE_HOST = "http://localhost"
INSIGHTFACE_PORT = 18081


class IFRClient:

    def __init__(self, host: str = INSIGHTFACE_HOST, port: str = INSIGHTFACE_PORT):
        self.server = f"{host}:{port}"
        self.session = requests.Session()

    def extract(self, data: list,
                mode: str = 'data',
                server: str = None,
                threshold: float = 0.6,
                extract_embedding=True,
                return_face_data=True,
                return_landmarks=True,
                embed_only=False,
                limit_faces=0,
                use_msgpack=True):

        if server is None:
            server = self.server

        extract_uri = f'{server}/extract'

        images = dict()
        if mode == 'data':
            images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)

        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   return_landmarks=return_landmarks,
                   embed_only=embed_only,  # If set to true, API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   use_rotation=True,
                   msgpack=use_msgpack,
                   )

        resp = self.session.post(extract_uri, json=req, timeout=120)
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)

        images = content.get('data')

        for im in images:
            status = im.get('status')

            if status != 'ok':
                print(content.get('traceback'))
                break

        return content

    def extract_faces_data(self, batch_frames: list, mode='data',
                          threshold=0.6, extract_embeddings=True, return_face_data=True, return_landmarks=False,
                          embed_only=False, limit_faces=0, use_msgpack=True):

        # Initialize an empty batch to store the encoded frame data
        batch_data = []

        for frame in batch_frames:
            # encode each frame in JPEG format
            _, buffer = cv2.imencode(".jpg", frame)
            # get base64 formatted data
            data = base64.b64encode(buffer.tobytes()).decode("ascii")
            batch_data.append(data)

        faces_data = self.extract(
            batch_data, mode=mode, server=self.server, threshold=threshold, extract_embedding=extract_embeddings,
            return_face_data=return_face_data, return_landmarks=return_landmarks, embed_only=embed_only,
            limit_faces=limit_faces, use_msgpack=use_msgpack
        )

        return faces_data

    def batch_face_locations(self, batch_of_frames: list, batch_of_cam_names: list, batch_of_cam_ips: list, mode='data',
                             threshold=0.6, extract_embeddings=True, return_face_data=True, return_landmarks=False,
                             embed_only=False, limit_faces=0, use_msgpack=True):

        # Initialize an empty batch to store the encoded frame data
        batch = []

        for frame in batch_of_frames:
            # encode each frame in JPEG format
            _, buffer = cv2.imencode(".jpg", frame)
            # get base64 formatted data
            data = base64.b64encode(buffer.tobytes()).decode("ascii")
            batch.append(data)

        faces_data = self.extract(batch, mode=mode, server=self.server, threshold=threshold,
                                  extract_embedding=extract_embeddings, return_face_data=return_face_data,
                                  return_landmarks=return_landmarks, embed_only=embed_only, limit_faces=limit_faces,
                                  use_msgpack=use_msgpack)

        batch_encoding_list = []
        batch_frame_list = []
        batch_cam_list = []
        batch_ip_list = []
        batch_bbox_list = []

        for i, faces in enumerate(faces_data['data']):
            for face_data in faces["faces"]:
                bbox = face_data["bbox"]
                batch_bbox_list.append(bbox)
                encoding = face_data["vec"]
                encoding_array = numpy.array(encoding)
                batch_encoding_list.append(encoding_array)
                batch_frame_list.append(batch_of_frames[i])
                batch_cam_list.append(batch_of_cam_names[i])
                batch_ip_list.append(batch_of_cam_ips[i])

        return batch_encoding_list, batch_bbox_list, batch_frame_list, batch_cam_list, batch_ip_list

    def extract_face_data(self, frame: numpy.ndarray, mode='data',
                  threshold=0.6, extract_embeddings=True, return_face_data=True, return_landmarks=False,
                  embed_only=False, limit_faces=0, use_msgpack=True):

        # encode each frame in JPEG format
        _, buffer = cv2.imencode(".jpg", frame)
        # get base64 formatted data
        data = base64.b64encode(buffer.tobytes()).decode("ascii")

        faces_data = self.extract([data], mode=mode, server=self.server, threshold=threshold,
                                  extract_embedding=extract_embeddings, return_face_data=return_face_data,
                                  return_landmarks=return_landmarks, embed_only=embed_only, limit_faces=limit_faces,
                                  use_msgpack=use_msgpack)

        return faces_data

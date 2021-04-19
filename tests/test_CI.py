from FaceIDLight.tools import FaceDetection, FaceRecognition
from FaceIDLight.camera import Camera
from FaceIDLight.demo import Demonstrator
import cv2
import pytest
import numpy as np
import json


@pytest.mark.face_detection
def test_single_face():
    detector = FaceDetection()
    img = cv2.imread('tests/data/img1.png')
    detection = detector.detect_faces(img)
    assert (json.load(open('tests/data/detections.json', 'r'))['img1'] == detection[0][1]).all()


@pytest.mark.face_detection
def test_multi_face():
    detector = FaceDetection()
    img = cv2.imread('tests/data/img3.png')
    detection = detector.detect_faces(img)
    assert len(detection) == 2


@pytest.mark.face_detection
def test_no_face():
    detector = FaceDetection()
    img = cv2.imread('tests/data/img2.png')
    detection = detector.detect_faces(img)
    assert not detection


@pytest.mark.face_alignment
def test_align():
    detector = FaceDetection()
    img = cv2.imread('tests/data/img1.png')
    detection = json.load(open('tests/data/detections.json', 'r'))['img1']
    face = detector.get_face(img, np.asarray(detection))
    face_target = cv2.imread('tests/data/face.png')
    assert (face == face_target).all()


@pytest.mark.face_recognition
def test_recognition():
    recognizer = FaceRecognition()
    face = cv2.imread('tests/data/face.png')
    emb = recognizer.get_emb(face[None])
    assert (emb[0] == json.load(open('tests/data/detections.json', 'r'))['face_emb'][0]).all()


@pytest.mark.camera
def test_camera():
    camera = Camera()
    demo = Demonstrator()
    demo.run()
    assert camera





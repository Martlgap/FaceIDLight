from FaceIDLight.tools import FaceDetection, FaceRecognition
import cv2
import pytest
import numpy as np
import json


@pytest.mark.face_detection
def test_single_face():
    detector = FaceDetection()
    img = cv2.imread("tests/data/img1.png")
    detection = detector.detect_faces(img)
    target_detection = json.load(open("tests/data/detections.json", "r"))["img1"]
    assert abs(sum(sum(np.asarray(detection[0][1]) - np.asarray(target_detection)))) < 1e-3


@pytest.mark.face_detection
def test_multi_face():
    detector = FaceDetection()
    img = cv2.imread("tests/data/img3.png")
    detection = detector.detect_faces(img)
    assert len(detection) == 2


@pytest.mark.face_detection
def test_no_face():
    detector = FaceDetection()
    img = cv2.imread("tests/data/img2.png")
    detection = detector.detect_faces(img)
    assert not detection


@pytest.mark.face_alignment
def test_align():
    detector = FaceDetection()
    img = cv2.imread("tests/data/img1.png")
    detection = json.load(open("tests/data/detections.json", "r"))["img1"]
    face = detector.get_face(img, np.asarray(detection))
    face_target = cv2.imread("tests/data/face.png")
    print(sum(sum(sum(face - face_target))))
    assert abs(sum(sum(sum(face - face_target)))) < 5


@pytest.mark.face_recognition
def test_recognition():
    recognizer = FaceRecognition()
    face = cv2.imread("tests/data/face.png")
    emb = recognizer.get_emb(face[None])
    target_emb = json.load(open("tests/data/detections.json", "r"))["face_emb"]
    assert abs(sum(sum(np.asarray(emb[0]) - np.asarray(target_emb[0])))) < 0.2

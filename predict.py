import cv2 as cv

# Loading Caffe model and prototxt
faceDetectProto_path = "model/face_detect.prototxt"
faceDetectModel_path = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel"

agePredictProto_path = "model/age_deploy.prototxt"
agePredictModel_path = "model/age_net.caffemodel"

genderProto_path = "model/gender_deploy.prototxt"
genderModel_path = "model/gender_net.caffemodel"


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(agePredictModel_path, agePredictProto_path)
genderNet = cv.dnn.readNet(genderModel_path, genderProto_path)
faceNet = cv.dnn.readNet(faceDetectModel_path, faceDetectProto_path)


def getFaceBox(net, frame, conf_threshold=0.7):
	frameOpencvDnn = frame.copy()
	frameHeight = frameOpencvDnn.shape[0]
	frameWidth = frameOpencvDnn.shape[1]
	blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

	net.setInput(blob)
	detections = net.forward()

	bboxes = []

	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > conf_threshold:
			x1 = int(detections[0, 0, i, 3] * frameWidth)
			y1 = int(detections[0, 0, i, 4] * frameHeight)
			x2 = int(detections[0, 0, i, 5] * frameWidth)
			y2 = int(detections[0, 0, i, 6] * frameHeight)
			bboxes.append([x1, y1, x2, y2])
			cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
	return frameOpencvDnn, bboxes


def predictAgeGender(file):
	padding = 20

	frame = cv.imread('./static/'+file)
	frameFace, bboxes = getFaceBox(faceNet, frame)

	if not bboxes:
		# print("No face Detected, Checking next frame")
		return "No Face Detected", "No Face Detected"

	for bbox in bboxes:

		face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

		blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
		genderNet.setInput(blob)
		genderPreds = genderNet.forward()
		gender = genderList[genderPreds[0].argmax()]

		#print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

		ageNet.setInput(blob)
		agePreds = ageNet.forward()
		age = ageList[agePreds[0].argmax()]

		#print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

		return gender, age






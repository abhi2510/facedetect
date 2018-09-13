import cv2
import sys

# Get user supplied values
#imagePath = sys.argv[1]
casc_face_Path = "haarcascade_frontalface_default.xml"
casc_eye_Path = "haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(casc_face_Path) #it cascade the frontal_face
eyeCascade =cv2.CascadeClassifier(casc_eye_Path)

cap = cv2.VideoCapture(0)					#it start the video via laptop camera

while True:
	ret,image =cap.read()				#read the image from camera
	#image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #covert the imge into gray like black n white
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,1.1,5) # give the dimension of face
		#print(faces)
	for (x, y, w, h) in faces:
		print("Found {0} faces!".format(len(faces)))						# Draw a rectangle around the faces
		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eye = eyeCascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eye:
			print("Found {0} eye!".format(len(eye)))
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh) ,(255,0,0),2)
		cv2.imshow("Faces found", image)
		k = cv2.waitKey(30) & 0xff
		if k==27:
			break

cap.release()
cv2.destroyAllWindows()	

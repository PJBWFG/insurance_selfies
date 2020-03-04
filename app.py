from flask import render_template, request, Flask, flash, url_for, redirect
from werkzeug.utils import secure_filename
import cv2 as cv
import argparse
import time
import os
from predict import predictAgeGender, getFaceBox

UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# App definition
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'many random bytes'



def predictAge():
	img = cv.imread('./static/temp.jpg')

	getFaceBox(faceNet, img)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 

@app.route('/', methods=['POST','GET'])
def predict():

	if request.method == 'GET':
		return render_template("index.html")

	if request.method == 'POST':

		if 'imageUpload' not in request.files:
			#flash('No file part')
			return redirect(request.url)

		file = request.files['imageUpload']

		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			#flash('No selected file')
			print("YES")
			return redirect(request.url)

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)

			for files in os.listdir('static/'):
				os.remove('static/' + files)

			new_filename = "temp" + str(time.time()) + ".jpg"

			file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
			gender, age = predictAgeGender(new_filename)
			#print("Age: ", age)
			#print("Gender: ", gender)
			return render_template("output.html", age=age, gender=gender, img_src=new_filename)

if __name__ == "__main__":
	app.run()

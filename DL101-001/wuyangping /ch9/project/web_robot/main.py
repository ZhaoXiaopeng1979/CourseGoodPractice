# -*- coding: utf-8 -*-

from flask import Flask, Response
from GenerateResponse import response
import time
from predict_images import predict_images
app = Flask(__name__)


@app.route('/')
def main():
    return 'Welcome to the AI world! I am a robot who can chat and recognize images. Have fun :)'

@app.route("/vision")
def vision():
    result = predict_images()
    return result

@app.route("/image/<imageid>")
def index(imageid):
    #if (imageid=='run'):
    #    result = predict_images()
    #    return result
    image = file("image/{}.jpg".format(imageid))
    print(imageid)
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route('/chat/<context>')
def chat(context):
	# show the user profile for that user
	#result = 'User: %s' % context
	#result += 'Robot: ' + response(context)
    t0 = time.time()
    output = response(context)
    t1 = time.time()
    duration = t1-t0
    return '[My input]: %s, [Robot response]:%s, Time Spent:%.3f s' % (context, output, duration)

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'Welcome, %s!' % username

if __name__ == '__main__':
    app.run()

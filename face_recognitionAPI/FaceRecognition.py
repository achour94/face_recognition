# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 22:17:35 2020

@author: achour
"""
from flask import Flask, jsonify, request, after_this_request

import face_recognition
import pickle
import numpy as np
import cv2
import base64
from PIL import Image

def add_person_func (img, name):
    with open('dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)
    image = face_recognition.load_image_file(img)
    image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    all_face_encodings[name] = face_recognition.face_encodings(image_cvt)[0]
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(all_face_encodings,f)


def load_dataset ():
    with open('dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)
    return  list(all_face_encodings.keys()),np.array(list(all_face_encodings.values()))

def get_Img_loc_enc ():
    #capture the video from default camera
    webcam_video_stream = cv2.VideoCapture(0)    
    while True:
        #get the current frame from the video stream as an image
        ret,current_frame = webcam_video_stream.read()
        #resize the current frame to 1/4 size to process faster
        current_frame_small = cv2.resize(current_frame, (0,0),fx=0.25,fy=0.25)
        #♣detect all faces in the image
        all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model="hog")
        if all_face_locations:
            webcam_video_stream.release()
            cv2.destroyAllWindows()
            break
    #detect face encodings for all the faces detected
    image_cvt = cv2.cvtColor(current_frame_small, cv2.COLOR_BGR2RGB)
    all_face_encodings = face_recognition.face_encodings(image_cvt,all_face_locations)
    return current_frame,all_face_locations, all_face_encodings 


def recognize_img (img, loc, enc,knownFaceNames, knownFaceEnc):
    #looping through the faces locations and the face embeddings
    for current_face_location,current_face_encoding in zip(loc,enc):
        #splitting the tuple to get the four position values
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        top_pos = top_pos * 4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(knownFaceEnc, current_face_encoding,tolerance=0.5)
        #string to hold name 
        name_of_person = 'inconnu'
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = knownFaceNames[first_match_index]
        
        #draw rectangle around the faces
        cv2.rectangle(img,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        #display the name 
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name_of_person, (left_pos,bottom_pos+ 40), font, 2, (255,255,255),3)
        ret, buffer = cv2.imencode('.jpg', img)
        imgConvert = base64.b64encode(buffer).decode()
    
    return imgConvert, name_of_person
    return img
      


face_names,face_encodings = load_dataset()
app = Flask(__name__)

@app.route("/", methods=['GET'])
def dummy_api():
    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    img, loc, enc = get_Img_loc_enc()
    imgR, nom = recognize_img(img, loc, enc, face_names, face_encodings)
    jsonrep = {'nom': nom, 
               'img': imgR}
    return jsonify(jsonrep)
@app.route("/admin", methods=["POST"])
def add_person():
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    name = request.form['name']
    file = request.files['image']
    print(request.form['name'])
    file.save(file.filename)
    add_person_func(file.filename, name)
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("Image", img)
    return jsonify({'msg': 'success'})

if __name__ == "__main__":
    app.run(host='0.0.0.0')



    


#cv2.imshow("faces identified", imgR)

   

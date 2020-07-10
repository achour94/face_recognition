# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 19:30:30 2020

@author: achour
"""

import cv2
import face_recognition


#load the image to recognize
original_image = cv2.imread('images/testing/aliach.jpg')


#load the sample image and get 128 face encoding from them
salim_image = face_recognition.load_image_file('images/samples/salim.jpg')
salim_face_encoding = face_recognition.face_encodings(salim_image, num_jitters=100)[0]

achour_image = face_recognition.load_image_file('images/samples/achour.jpg')
achour_face_encoding = face_recognition.face_encodings(achour_image, num_jitters=100)[0]

#save the encodings and the corresponding labels in separate arrays in the same order
known_face_encoding = [salim_face_encoding, achour_face_encoding]
known_face_names = ["salim hassouna", "achour berrahma"]

#load the unkown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file('images/testing/mha.jpg')


#detect all faces
all_face_locations = face_recognition.face_locations(image_to_recognize, model="hog")
#detect face encodings for all the faces detected
all_face_encodings = face_recognition.face_encodings(image_to_recognize,all_face_locations)

#print the number of faces detected
print('there are {} no of faces in this img'.format(len(all_face_locations)))

#looping through the faces locations and the face embeddings
for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
    #splitting the tuple to get the four position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    #find all the matches and get the list of matches
    all_matches = face_recognition.compare_faces(known_face_encoding, current_face_encoding)
    #string to hold name 
    name_of_person = 'uknown face'
    #check if the all_matches have at least one item
    #if yes, get the index number of face that is located in the first index of all_matches
    #get the name corresponding to the index number and save it in name_of_person
    if True in all_matches:
        first_match_index = all_matches.index(True)
        name_of_person = known_face_names[first_match_index]
    #draw rectangle around the faces
    cv2.rectangle(original_image,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
    
    #display the name 
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(original_image, name_of_person, (left_pos,bottom_pos+ 40), font, 2, (255,255,255),3)
    
    #display the image
    imgR = cv2.resize(original_image, (960,540))
    cv2.imshow("faces identified", imgR)
import streamlit as st
from PIL import Image
import tensorflow as tf
import os, json, math, librosa
import numpy as np

img = Image.open("/Users/apoorva/Desktop/images.jpeg")
st.image(img,width=700)
st.title("MUSIC GENRE CLASSIFIER")

st.markdown("We have developed music genre classifier using Convolutional Neutral Network(CNN), It will take the input .wav file and classify it into its respective genre")

st.markdown("There are 10 genre as follows :")
genre_list = ['Blue','Country','Disco','Hiphop','Jazz','Metal','Pop','Reggae','Rock','Classical']

for i in genre_list:
	st.markdown('**'+i+'**')

st.write("Our input will be a 30 seconds audio file like")
audio_file = open('/Users/apoorva/Desktop/CODES/Data/genres_original/disco/disco.00002.wav','rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes,format='audio/wav')

st.markdown("**_Upload the input file (.wav)_**")

audio_file = st.file_uploader("Upload the audio:",type=["wav"])
if audio_file is None:
	st.error("Upload a file!!")
	
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/Users/apoorva/Desktop/CODES/Notebook/music_model.hdf5')
  return model

with st.spinner('Model is being loaded..'):
  model=load_model()

#DATASET_PATH = audio_file
JSON_PATH = "music.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(file_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

   
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                
                    
    for d in range(num_segments):
        start = samples_per_segment * d
        finish = start + samples_per_segment

                        
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

                       
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            #data["labels"].append(i-1)
            

    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if audio_file is not None:
    save_mfcc(audio_file, JSON_PATH, num_segments=6)
else:
	st.warning("Upload a file!!")


DATA_PATH = "./music.json"


def load_data(data_path):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    #y = np.array(data["labels"])
    z = np.array(data['mapping'])
    return X,z

X,z = load_data(DATA_PATH)

X = X[..., np.newaxis]

def predict(model, X):


    
    prediction = model.predict(X)

    predicted_index = np.argmax(prediction, axis=1)
    
    
    #predicted = z[predicted_index]

    #print("Predicted label: {}".format(predicted))

    return prediction

p = predict(model,X)

music_dict = {0:'Pop',1:'Metal',2:'Disco',3:'Blues',4:'Reggae',5:'Classical',6:'Rock',7:'HipHop',8:'Country',9:'Jazz'}

for key,value in music_dict.items():

    if p[0][key]>0.7:
        st.success("Predicted Genre :" + value)
        st.info("Predicted Accuracy: {}".format(p[0][key]))


st.markdown("<h6 style='text-align: center;'>Made by</h6>", unsafe_allow_html=True)
st.markdown("<h6>Shubh Almal &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Riya Gupta &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; Apoorva Reddy</h6>", unsafe_allow_html=True)
st.markdown("<h6><strong>19BCE2130 </strong> &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;<strong> 19BCE2072 </strong>&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;<strong> 19BCE2196 </strong></h6>", unsafe_allow_html=True)





# if p[0][0]>0.7:
# 	st.success("Predicted Genre : Pop")
# 	st.info("Predicted Accuracy: {}".format(p[0][0]))
# elif p[0][1]>0.7:
# 	st.success("Predicted Genre : Metal")
# 	st.info("Predicted Accuracy: {}".format(p[0][1]))
# elif p[0][2]>0.7:
# 	st.success("Predicted Genre : Disco")
# 	st.info("Predicted Accuracy: {}".format(p[0][2]))
# elif p[0][3]>0.7:
# 	st.success("Predicted Genre : Blues")
# 	st.info("Predicted Accuracy: {}".format(p[0][3]))
# elif p[0][4]>0.7:
# 	st.success("Predicted Genre : Reggae")
# 	st.info("Predicted Accuracy: {}".format(p[0][4]))
# elif p[0][5]>0.7:
# 	st.success("Predicted Genre : Classical")
# 	st.info("Predicted Accuracy: {}".format(p[0][5]))
# elif p[0][6]>0.7:
# 	st.success("Predicted Genre : Rock")
# 	st.info("Predicted Accuracy: {}".format(p[0][6]))
# elif p[0][7]>0.7:
# 	st.success("Predicted Genre : HipHop")
# 	st.info("Predicted Accuracy: {}".format(p[0][7]))
# elif p[0][8]>0.7:
# 	st.success("Predicted Genre : Country")
# 	st.info("Predicted Accuracy: {}".format(p[0][8]))
# elif p[0][9]>0.7:
# # 	st.success("Predicted Genre : Jazz")
# 	st.info("Predicted Accuracy: {}".format(p[0][9]))









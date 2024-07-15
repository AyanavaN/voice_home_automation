import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 
import subprocess
import sys
from vosk import Model, KaldiRecognizer, SetLogLevel
warnings.filterwarnings("ignore")
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

TEST_WAV_DIR = "testing_set"
TRAIN_WAV_DIR = "training_set"
MODEL_DIR = "trained_models"
TRAIN_DATA = []
TEST_DATA = []
NO_OF_SAMPLES = 2

MODELS_LOADED = []

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 3
def calculate_delta(array):
   
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined

def saveWave(sample_size,filename="test.wav",dir=".",frames=()):
	RecordFrames = frames
	OUTPUT_FILENAME = filename
	WAVE_OUTPUT_FILENAME = os.path.join(dir, OUTPUT_FILENAME)
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(sample_size)
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(RecordFrames))
	waveFile.close()
def getAudio():
	device_index = 2
	audio = pyaudio.PyAudio()
	print("----------------------record device list---------------------")
	info = audio.get_host_api_info_by_index(0)
	numdevices = info.get('deviceCount')
	for i in range(0, numdevices):
		if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
			print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
	print("-------------------------------------------------------------")
	index = int(input())
	print("recording via index " + str(index))
	stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True, input_device_index=index,
						frames_per_buffer=CHUNK)
	print("recording started")
	Recordframes = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		Recordframes.append(data)
	print("recording stopped")
	stream.stop_stream()
	stream.close()
	audio.terminate()
	return  (tuple(Recordframes),audio.get_sample_size(FORMAT))

def TrainModel():
	ans = "y"
	while ans.lower() == "y" :
		speaker_id =(input("Please Enter Your Name:"))
		TRAIN_DATA.append(speaker_id)
		for cnt in range(NO_OF_SAMPLES):
			Frames,sample_size = getAudio()
			saveWave(sample_size,speaker_id+str(cnt),TRAIN_WAV_DIR,Frames)
		ans = input("Take another input (Y|N)? ")
		Training()
		# saving authorized name
		file = open("myfiles.txt", "a")
		file.write(speaker_id + "\n")
		file.close()
def TestModel():
	FILENAME = "test.wav"
	Frames,sample_size=getAudio()
	saveWave(sample_size,FILENAME,TEST_WAV_DIR,Frames)

	Testing()
def Training():

	source   = "training_set"
	dest = "trained_models"
	features = np.asarray(())
	for speaker in TRAIN_DATA:
		for i in range(NO_OF_SAMPLES):
			sr,audio = read(os.path.join(source,speaker+str(i)))
			vector = extract_features(audio,sr)
			if i == 0:
				features = vector
			else :
				features = np.vstack((features, vector))
		gmm = GaussianMixture(n_components=NO_OF_SAMPLES, max_iter=200, covariance_type='diag', n_init=2)
		gmm.fit(features)

		# dumping the trained gaussian model
		picklefile = speaker + ".gmm"
		pickle.dump(gmm, open(os.path.join(dest , picklefile), 'wb'))
		print('+ modeling completed for speaker:', picklefile, " with data point = ", features.shape)
		features = np.asarray(())
	TRAIN_DATA.clear()


def Testing():
	source   = "./testing_set"
	modelpath = "./trained_models/"
	gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath)]
	#Load the Gaussian gender Models
	modLen = len(MODELS_LOADED)
	true_speaker = os.path.basename(modelpath)

	if(modLen == 0):
		models  = [pickle.load(open(fname,'rb')) for fname in gmm_files]
		modLen = len(models)
	else:
		models = MODELS_LOADED
	if len(TRAIN_DATA) == 0 :
		data  = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]
	else:
		data = TRAIN_DATA
	sr,audio = read(os.path.join(source,"test.wav"))
	vector   = extract_features(audio,sr)
	log_likelihood = np.zeros(modLen)
	for i in range(modLen):
		gmm = models[i]  #checking with each model one by one
		scores = np.array(gmm.score(vector))
# 		print("Individual speaker scores: ", scores)
		log_likelihood[i] = int(scores.sum())
		winner = np.argmax(log_likelihood)
		winnerRange = log_likelihood[winner]
	if(winnerRange > -27):
		print("\tdetected as - ", data[winner])
	else:
		print("Unknown")

	time.sleep(1.0)
	print("Indidvidual log likelihoods: ", log_likelihood)
	#storing the verified names in verify
	f = open("myfiles.txt", "r")
	verify=f.read()
	# print(verify)

	



# SPEECH TO TEXT CONVERSION of authentic user
	if data[winner] in verify:
		SetLogLevel(0)

		model = Model("Vosk/vosk-model-en-in-0.5/")
		rec = KaldiRecognizer(model, RATE)


		res1=""
		with open("testing_set/test.wav", "rb") as f:
			data = f.read()
			rec.AcceptWaveform(data)
			res1=rec.FinalResult()
		result = res1.split(" : ")[1].split("\n")[0].replace('"','')
		print(result)
		# if 'kitchen' in result:
		# 	if 'light' in result:
		# 		if 'on' in result:
		# 			print("kitchen light is on")
		# if 'kitchen' in result:
		# 	if 'light' in result:
		# 		if 'off' in result:
		# 			print("kitchen light is off")
		# if 'bedroom' in result:
		# 	if 'light' in result:
		# 		if 'on' in result:
		# 			print("bedroom light is on")
		if 'light' in result:
			if 'on' in result:
				print("turning the led onn")
				GPIO.output(11, GPIO.LOW)
		if 'light' in result:
			if 'off' in result:
				print("turning the led off")
				GPIO.output(11, GPIO.HIGH)


# # 		led.off()
# 	if 'kitchen' and 'lights' and 'on' in result:
# 		print("Lights on!")
# # 		led.on()
# 	if(winnerRange < -27):
# 		print("Speaker Unidentified")
# # 		led.off()






if __name__ == "__main__" :
	while True:
		choice = int(
			input("\n 1.TRAIN \n 2.TEST \n 3.EXIT \n"))
		if (choice == 1):
			TrainModel()
		elif (choice == 2):
			TestModel()
		elif(choice == 3):
			break
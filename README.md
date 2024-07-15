# voice_home_automation
Secured home Automation with Voice Recognition Using Python and Vosk

#Instruction:-

1) Download the vosk model into the vosk folder. You can find the link inside the vosk folder.

2) Install Required Packages by executing the following code

3) Execute pip install -r requirements.txt

4) Then execute the new2 python file.

5) If you are on a raspberry pi you can un-comment the GPIO part.




Project Description: Home Automation Using Voice Recognition

Introduction:-

The Home Automation Using Voice Recognition project aims to create an intelligent system that allows users to control various home appliances using their voice. This system leverages advanced speech recognition technologies and machine learning algorithms to provide a seamless and hands-free experience for managing home environments.


Objectives:-

Voice-Controlled Home Automation: Enable users to control home appliances such as lights, fans, and other electronic devices using voice commands.

User Authentication: Implement a voice recognition mechanism to authenticate users and ensure that only authorized individuals can control the home appliances.

Speech-to-Text Conversion: Convert voice commands into text to interpret and execute the corresponding actions.



Key Features:-

Voice Command Recognition: The system can accurately recognize and interpret voice commands given by the user.

User Authentication: Each user is required to train the system with their voice samples, which are used to create unique voice models for authentication.

Control Home Appliances: The system can turn on/off lights and other appliances based on the recognized voice commands.

Real-Time Processing: The system processes voice commands in real-time, providing immediate feedback and action.



Technical Details:-


1)Hardware Components:-

Microphone: To capture the user's voice commands.

Raspberry Pi: For interfacing with home appliances and running the voice recognition software.

Relays: To control the electrical appliances.


2)Software Components:-

Python: The primary programming language used to implement the system.

pyaudio: For capturing audio from the microphone.

wave: For handling WAV audio files.

scikit-learn: For implementing the Gaussian Mixture Models (GMM) used in user authentication.

Vosk: An open-source speech recognition toolkit for converting speech to text.

python_speech_features: For extracting MFCC features from audio signals.


3)Voice Recognition Process:-

Training Phase: Users provide voice samples which are processed to extract Mel Frequency Cepstral Coefficients (MFCC) features. These features are used to train Gaussian Mixture Models (GMM) for each user.

Testing Phase: When a user issues a voice command, the system captures the audio, extracts MFCC features, and compares them against the trained GMM models to authenticate the user.

Command Execution: Once authenticated, the system uses Vosk to convert the voice command into text and executes the corresponding action (e.g., turning on/off a light).



Usage:-

1)Training the System: Users need to train the system by providing voice samples. This process involves recording a few samples of their voice, which are then used to create a unique voice model for each user.

2)Issuing Commands: To control an appliance, the user simply speaks a command (e.g., "Turn on the light"). The system recognizes the command, authenticates the user, and executes the action.

3)Feedback: The system provides real-time feedback, indicating whether the command was successfully executed or if there was an error (e.g., unrecognized command or unauthorized user).

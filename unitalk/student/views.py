from django.shortcuts import render, redirect
from django.http import HttpResponse 
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.http import require_POST
from .sign_language_recognizer import SignLanguageRecognizer
from django.contrib.auth import authenticate, login
from .models import Student
from django.contrib import messages
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import base64
import json
from keras.models import model_from_json
from string import ascii_uppercase
import operator
from spellchecker import SpellChecker
import os
import uuid
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model

# Create your views here.
def home(request):
    return render(request, 'home.html', {'user': request.user})

def about(request):
    # return HttpResponse('This is about page')
    return render(request, 'about.html')

from django.contrib.auth.hashers import make_password  # Import make_password

def add_student(request):
    s = Student()
    s.student_name = request.POST.get('name')
    s.username = request.POST.get('username')
    s.mobile_no = request.POST.get('mobile')
    s.email_id = request.POST.get('email')
    s.password = request.POST.get('password')
    s.confirm_password = request.POST.get('confirmPassword')
    s.save()
    # print(s.mobile_no)
    return redirect('login')
    
    
def register(request):
    return render(request, 'register.html')
    

def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        try:
            student = Student.objects.get(username=username, password=password)
            # Here, set the session or authentication as needed
            messages.success(request, 'Login successful!')
            return redirect('home')  # Redirect to a home page or dashboard
        except Student.DoesNotExist:
            messages.error(request, 'Invalid username or password')
            return redirect('login')

    return render(request, 'login.html')

def logout(request):
    return render(request, 'logout.html')
    

def course(request):
    return render(request, 'course.html')

def learning(request):
    return render(request, 'learning.html')

def isl(request):
    return render(request, 'isl.html')

# List of colors for visualization (optional)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117), (117, 16, 245),
          (16, 245, 117), (245, 245, 16), (16, 245, 245), (245, 16, 245), (16, 16, 245),
          (245, 245, 245), (117, 117, 117), (16, 16, 16), (245, 117, 117), (117, 245, 117),
          (117, 117, 245), (245, 16, 16), (16, 245, 16), (16, 16, 245), (245, 245, 16),
          (16, 245, 245), (245, 16, 245), (117, 16, 245), (245, 245, 117), (117, 245, 245),
          (245, 117, 245), (245, 16, 117), (16, 245, 117), (117, 16, 245), (16, 117, 245),
          (117, 245, 16), (245, 117, 16), (16, 117, 245), (245, 16, 117), (117, 16, 245),
          (16, 245, 117), (117, 245, 16)]

# Load the main LSTM model
model_path_main = r'D:\Final Year project\Project\unitalk\student\models\final3.h5'
model_main = load_model(model_path_main)

# Load additional LSTM models for confusing letter groups
model_path_ABG = r'D:\Final Year project\Project\unitalk\student\models\model_ABG.h5'
model_ABG = load_model(model_path_ABG)

model_path_DPYT = r'D:\Final Year project\Project\unitalk\student\models\model_DPYT.h5'
model_DPYT = load_model(model_path_DPYT)

model_path_CVL = r'D:\Final Year project\Project\unitalk\student\models\model_CLV.h5'
model_CVL = load_model(model_path_CVL)

model_path_OIU = r'D:\Final Year project\Project\unitalk\student\models\model_OIU.h5'
model_OIU = load_model(model_path_OIU)

model_path_EFJ = r'D:\Final Year project\Project\unitalk\student\models\model_EFJ.h5'
model_EFJ = load_model(model_path_EFJ)

model_path_MNRW = r'D:\Final Year project\Project\unitalk\student\models\model_MNRW.h5'
model_MNRW = load_model(model_path_MNRW)

# Define the sequence length
SEQ_LENGTH = 30
input_sequences = []
MIN_LETTER_COUNT_FOR_WORD = 3

# Define the actions for each model
actions_main = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J','L', 'M', 'N', 'O', 'P', 'R', 'S','T', 'U', 'V', 'W', 'Y'])
actions_ABG = np.array(['A', 'B', 'G'])
actions_DPYT = np.array(['D', 'P', 'Y', 'T'])
actions_CVL = np.array(['C', 'V', 'L'])
actions_OIU = np.array(['O', 'I', 'U'])
actions_EFJ = np.array(['E', 'F', 'J'])
actions_MNRW = np.array(['M', 'N', 'R', 'W'])

# Dictionary mapping confusing letter groups to their corresponding models and actions
group_model_mapping = {
    'ABG': (model_ABG, actions_ABG),
    'DPYT': (model_DPYT, actions_DPYT),
    'CVL': (model_CVL, actions_CVL),
    'OIU': (model_OIU, actions_OIU),
    'EFJ': (model_EFJ, actions_EFJ),
    'MNRW': (model_MNRW, actions_MNRW)
}

# Define the confusing groups
confusing_groups = {
    'A': 'ABG', 'B': 'ABG', 'G': 'ABG',
    'D': 'DPYT', 'P': 'DPYT', 'Y': 'DPYT', 'T': 'DPYT',
    'C': 'CVL', 'V': 'CVL', 'L': 'CVL',
    'O': 'OIU', 'I': 'OIU', 'U': 'OIU',
    'E': 'EFJ', 'F': 'EFJ', 'J': 'EFJ',
    'M': 'MNRW', 'N': 'MNRW', 'R': 'MNRW', 'W': 'MNRW'
}

# Define a global variable to store the sequence of predicted letters
predicted_sequence = []

# Define a global variable to store the previous predicted letter
previous_predicted_letter = None

# Initialize Mediapipe Holistic once and reuse it
mp_holistic = mp.solutions.holistic.Holistic()

def preprocess_frame(frame):
    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set image to non-writeable
    frame_rgb.flags.writeable = False

    # Make prediction
    results = mp_holistic.process(frame_rgb)

    # Set image to writeable
    frame_rgb.flags.writeable = True

    # Convert image back to BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Extract keypoints
    keypoints = extract_keypoints(results)

    return keypoints

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Load the dictionary from the JSON file
with open(r'D:\Final Year project\Project\unitalk\student\words_dictionary.json', 'r') as file:
    sign_language_dictionary = json.load(file)

# Initialize spell checker
spell_checker = SpellChecker()

def generate_words(predicted_sequence):
    matched_words = []

    # Compare the predicted sequence with words in the dictionary
    for word, sequence in sign_language_dictionary.items():
        if sequence == predicted_sequence:
            matched_words.append(word)

    # Filter out words below a minimum length threshold
    min_word_length = 3
    matched_words = [word for word in matched_words if len(word) >= min_word_length]

    # Prioritize longer words over shorter ones
    matched_words.sort(key=len, reverse=True)

    # Apply spelling correction to the matched words
    corrected_words = [spell_checker.correction(word) for word in matched_words]

    return corrected_words


def predict_hand_sign(request):
    global input_sequences, previous_predicted_letter, predicted_sequence

    if request.method == 'GET':
        return render(request, 'recognize_sign_language.html')
    elif request.method == 'POST':
        if 'reset_sequence' in request.POST:  # Check if the reset button was pressed
            # Clear the predicted sequence
            predicted_sequence = []
            return JsonResponse({'status': 'Sequence reset successfully'})
        
        frame_data = request.POST.get('frame')
        if frame_data:
            # Convert base64 encoded image to numpy array
            encoded_data = frame_data.split(',')[1]
            decoded_data = base64.b64decode(encoded_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame (convert to sequence of keypoints)
            keypoints = preprocess_frame(frame)
            
            # Append the keypoints to the global variable to accumulate sequences
            input_sequences.append(keypoints)
            
            # Check if enough frames have been accumulated to form a sequence
            if len(input_sequences) == SEQ_LENGTH:
                # Take the last SEQ_LENGTH frames to form a sequence
                input_sequence = input_sequences[-SEQ_LENGTH:]
                
                # Pass the sequence to the main LSTM model for prediction
                prediction_main = model_main.predict(np.array([input_sequence]))
                predicted_text_main = actions_main[np.argmax(prediction_main)]
                
                # If the predicted letter belongs to a confusing group, use the corresponding model for re-prediction
                if predicted_text_main in confusing_groups:
                    group = confusing_groups[predicted_text_main]
                    group_model, group_actions = group_model_mapping[group]
                    prediction_group = group_model.predict(np.array([input_sequence]))
                    predicted_text_group = group_actions[np.argmax(prediction_group)]
                    final_prediction = predicted_text_group
                else:
                    final_prediction = predicted_text_main
                
                # Check if the predicted letter has changed
                if final_prediction != previous_predicted_letter:
                    predicted_sequence.append(final_prediction)
                    previous_predicted_letter = final_prediction

                    # Generate words from the predicted sequence
                    words = generate_words(predicted_sequence)
                
                else:
                    words = []  # Initialize words variable if the predicted letter hasn't changed
                
                # Clear the input sequences after each prediction
                input_sequences = []
                print(words)

                # Construct the JSON response with the predicted text and sequence
                response_data = {'predicted_letter': final_prediction, 'predicted_sequence': predicted_sequence, 'words': words}
                return JsonResponse(response_data)
            else:
                return JsonResponse({'status': 'Waiting for more frames'})
        else:
            return JsonResponse({'error': 'No frame data provided'})
    else:
        return JsonResponse({'error': 'Invalid request method'})



# Ensure that resources are properly released when the server shuts down
import atexit
@atexit.register
def cleanup():
    mp_holistic.close()

# # Function to generate words from predicted letters
# def generate_words(predicted_letters):
#     words = []
#     word = ''
#     letter_count = 0
#     for letter in predicted_letters:
#         if letter_count < MIN_LETTER_COUNT_FOR_WORD:
#             # If the minimum letter count for a word is not reached, continue adding letters to the current word
#             word += letter
#             letter_count += 1
#         else:
#             # If the minimum letter count for a word is reached, add the word to the list of words and start a new word
#             words.append(word)
#             word = letter
#             letter_count = 1
#     words.append(word)  # Add the last word
#     return words

# # Function to suggest words using spellchecker
# def suggest_words(words):
#     suggestions = {}
#     spell = SpellChecker()
#     for word in words:
#         if not spell.correction(word) == word:
#             # If the word is misspelled, suggest corrections
#             suggestions[word] = spell.candidates(word)
#     return suggestions

# # Function to extract meaningful words from the sequence
# def extract_meaningful_word(words):
#     for word in words:
#         # Add logic to determine if word is meaningful
#         if len(word) >= MIN_LETTER_COUNT_FOR_WORD:
#             return word
#     return None
# Define your model and actions
actions = np.array(['A', 'B', 'G'])
model_path_evaluate = r'D:\Final Year project\Project\unitalk\student\models\model_ABG.h5'
model_evaluate = load_model(model_path_evaluate)

# Initialize input_sequences and SEQ_LENGTH
input_sequences = []
SEQ_LENGTH = 30

def model_evaluation(request):
    global input_sequences  # Ensure we're modifying the global input_sequences variable
    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == 'POST':
        frame_data = request.POST.get('frame')
        if frame_data:
            # Convert base64 encoded image to numpy array
            encoded_data = frame_data.split(',')[1]
            decoded_data = base64.b64decode(encoded_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the frame (convert to sequence of keypoints)
            keypoints = preprocess_frame(frame)
            
            # Append the keypoints to the input sequence
            input_sequences.append(keypoints)
            
            # Check if enough frames have been accumulated to form a sequence
            if len(input_sequences) == SEQ_LENGTH:
                # Take the last SEQ_LENGTH frames to form a sequence
                input_sequence = input_sequences[-SEQ_LENGTH:]
                
                # Pass the sequence to the model for prediction
                prediction = model_evaluate.predict(np.array([input_sequence]))
                predicted_letter = actions[np.argmax(prediction)]
                print(predicted_letter)
                # Clear the input_sequences after each prediction
                input_sequences = []
                
                response_data = {'predicted_letter': predicted_letter}
                return JsonResponse(response_data)
            else:
                return JsonResponse({'status': 'Waiting for more frames'})
        else:
            return JsonResponse({'error': 'No frame data provided'})
    else:
        return JsonResponse({'error': 'Invalid request method'})




# def generate_word_sequence(predicted_text):
#     word_sequence = []
#     word = ''
#     word_count = {}
#     for letter in predicted_text:
#         # Count the frequency of each letter
#         if letter not in word_count:
#             word_count[letter] = 1
#         else:
#             word_count[letter] += 1
#         # Add letter to word sequence if it has been predicted a certain number of times
#         if word_count[letter] >= MIN_LETTER_COUNT_FOR_WORD:
#             word += letter
#             # Check if the formed word is a meaningful word
#             if word in spell:
#                 word_sequence.append(word)
#                 word = ''
#                 word_count = {}
#     # Get word suggestions for the remaining letters
#     remaining_letters = ''.join(predicted_text[len(''.join(word_sequence)):])
#     word_suggestions = spell.candidates(remaining_letters)
#     return {'word_sequence': word_sequence, 'word_suggestions': word_suggestions}




# class SignLanguageRecognizer:
#     def __init__(self):
#         self.directory = 'D:/Final Year project/Project/unitalk/student/model'
#         self.spell_checker = SpellChecker()
#         self.vs = cv2.VideoCapture(0)
#         self.current_image = None
#         self.current_image2 = None
#         self.str = ""
#         self.blank_flag = 0
#         self.word = ""
        
#         # Load models
#         self.loaded_model = self.load_model("atoz")
#         self.loaded_model_dru = self.load_model("model-bw_dru")
#         self.loaded_model_tkdi = self.load_model("model-bw_tkdi")
#         self.loaded_model_smn = self.load_model("model-bw_smn")
        
#         # Initialize letter count
#         self.ct = {'blank': 0}
#         for letter in ascii_uppercase:
#             self.ct[letter] = 0

#         print("Loaded models from disk")

#     def load_model(self, model_name):
#         json_file = open(os.path.join(self.directory, f"{model_name}.json"), "r")
#         model_json = json_file.read()
#         json_file.close()
#         loaded_model = model_from_json(model_json)
#         loaded_model.load_weights(os.path.join(self.directory, f"{model_name}.h5"))
#         return loaded_model

#     def process_frames(self, frame):
#         print("Processing frames...")
#         cv2image = cv2.flip(frame, 1)
#         x1 = int(0.5 * frame.shape[1])
#         y1 = 10
#         x2 = frame.shape[1] - 10
#         y2 = int(0.5 * frame.shape[1])
#         cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0), 1)
#         cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
#         self.current_image = Image.fromarray(cv2image)
#         cv2image = cv2image[y1:y2, x1:x2]
#         gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 2)
#         th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#         ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
#         # Resize the processed image to 128x128 pixels
#         resized_image = cv2.resize(res, (128, 128))

#         # Save the processed image using OpenCV
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         processed_image_filename = f"processed_image_{timestamp}.jpg"
#         processed_image_dir = 'D:/Final Year project/Project/unitalk/student/images'
#         os.makedirs(processed_image_dir, exist_ok=True)  # Ensure the directory exists or create it
#         processed_image_path = os.path.join(processed_image_dir, processed_image_filename)
#         cv2.imwrite(processed_image_path, resized_image)

#         predicted_text = self.predict(resized_image)
#         print("Predicted text:", predicted_text)
#         return predicted_text

#     def predict(self, test_image):
#         print("Predicting...")
#         print("Test image shape:", test_image.shape)
#         test_image = cv2.resize(test_image, (128, 128))
#         print("Test image shape:", test_image.shape)
        
#         # Initialize current_symbol with a default value
#         current_symbol = 'A'  # Choose a default symbol (could be any valid symbol)
        
#         # Reset self.str to an empty string
#         self.str = ""
        
#         # Perform predictions using loaded models
#         result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
#         result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
#         result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
#         result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
        
#         # Concatenate all prediction results
#         combined_result = np.concatenate((result, result_dru, result_tkdi, result_smn), axis=1)
        
#         # Create a list of letters from 'A' to 'Z' and 'blank'
#         letters = list(ascii_uppercase) + ['blank']
        
#         # Initialize a dictionary to store predictions
#         prediction = dict(zip(letters, combined_result[0]))
        
#         # Sort predictions in descending order
#         prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
#         # Get the most probable symbol
#         current_symbol = prediction[0][0]
        
#         # Update self.str whenever a valid symbol is obtained
#         if current_symbol != 'blank':
#             self.str = current_symbol
        
#         # Handle blank symbol and update self.word
#         if current_symbol == 'blank':
#             # Reset letter counts
#             for letter in ascii_uppercase:
#                 self.ct[letter] = 0
#             self.ct['blank'] = 0
            
#             # Process self.word if it's not empty
#             if self.blank_flag == 0 and len(self.word) > 0:
#                 self.blank_flag = 1
#                 self.str += " " + self.word
#                 self.word = ""
#         else:
#             # Update self.word and reset blank_flag
#             if len(self.str) > 16:
#                 self.str = ""
#             self.blank_flag = 0
#             self.word += current_symbol
#             print("Updated self.word:", self.word)
        
#         return self.str
from flask import Flask, request, jsonify
import base64
import pickle
import numpy as np
import librosa
from io import BytesIO
import xmlrunner 

app = Flask(__name__)

with open("model/vgg_model.pkl", "rb") as f:
    vgg_model = pickle.load(f)

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def predict_genre(audio_data, clf):
    signal, rate = librosa.load(BytesIO(audio_data))
    hop_length = 512
    n_fft = 2048
    n_mels = 128
    S = librosa.feature.melspectrogram(y=signal, sr=rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = S_DB.flatten()[:1200]

    # Predict the genre
    genre_label = clf.predict([S_DB])[0]
    return genres[genre_label]

# Classe pour l'erreur de traitement
class AudioProcessingError(Exception):
    pass

@app.route('/predict_vgg', methods=['POST'])
def predict_vgg():
    try:
        # Get the audio data from the request
        data = request.json.get('wav_music')
        if not data:
            return jsonify({'error': 'Aucun fichier audio fourni'}), 400
        
        # Decode the Base64 audio data
        audio_data = base64.b64decode(data)
        
        # Call the predict_genre function with the decoded audio data
        predicted_genre = predict_genre(audio_data, vgg_model)
        
        return jsonify({'genre': predicted_genre})
    
    except AudioProcessingError:
        return jsonify({'error': 'Erreur de traitement du fichier audio'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)




# import os
# import numpy as np
# import base64
# from io import BytesIO
# from flask import Flask, request, jsonify
# from pydub import AudioSegment
# from flask_cors import CORS
# import logging
# import tensorflow as tf

# # Disable GPU (use CPU only)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# logging.basicConfig(level=logging.DEBUG)
# base_path = os.path.abspath(os.path.dirname(__file__))

# tflite_model_path = os.path.join(base_path, "vgg_model.tflite")

# try:
#     interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
#     interpreter.allocate_tensors()
#     logging.info("TFLite model loaded successfully.")
# except Exception as e:
#     logging.error(f"Error loading TFLite model: {e}")
#     interpreter = None  # Set interpreter to None if it fails to load

# # Genre labels for classification (ensure these match your model's training labels)
# genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

# def preprocess_audio(audio_segment):
#     """
#     Preprocess audio to match the input shape expected by the model.
#     """
#     # Convert the audio segment to a numpy array (samples are in int16 format)
#     audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

#     # Get the sample rate
#     sr = audio_segment.frame_rate

#     # Normalize audio data
#     audio_data = audio_data / np.max(np.abs(audio_data))

#     # Make sure the audio is 150x150 (model's expected size)
#     # You can either trim or pad the audio
#     if len(audio_data) < 150 * 150:
#         padding = 150 * 150 - len(audio_data)
#         audio_data = np.pad(audio_data, (0, padding), 'constant')
#     else:
#         audio_data = audio_data[:150 * 150]

#     # Reshape the audio data to match the input shape expected by the model
#     audio_data = audio_data.reshape((1, 150, 150, 1))

#     return audio_data

# def predict_genre_with_tflite(features):
#     """
#     Predict the genre using the TensorFlow Lite model.
#     """
#     try:
#         # Set the input tensor
#         input_details = interpreter.get_input_details()
#         output_details = interpreter.get_output_details()
#         logging.debug(f"Input details: {input_details}")
#         logging.debug(f"Output details: {output_details}")

#         interpreter.set_tensor(input_details[0]['index'], features)
#         interpreter.invoke()

#         # Get the prediction output
#         output_data = interpreter.get_tensor(output_details[0]['index'])

#         # Get the predicted genre (label with the highest probability)
#         predicted_genre = genre_labels[np.argmax(output_data)]

#         return predicted_genre
#     except Exception as e:
#         logging.error(f"Error in TFLite prediction: {str(e)}")
#         return None

# @app.route('/classify', methods=['POST'])
# def classify_music():
#     try:
#         data = request.json

#         if 'wav_music' not in data:
#             return jsonify({'error': 'No audio data provided'}), 400

#         # Decode the base64-encoded audio data
#         try:
#             audio_data = base64.b64decode(data['wav_music'])
#         except Exception as e:
#             logging.error(f"Error decoding base64 audio data: {str(e)}")
#             return jsonify({'error': 'Error during prediction'}), 400  # More generic error message

#         # Attempt to load the audio data using pydub
#         try:
#             audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')
#         except Exception as e:
#             logging.error(f"Error loading audio data: {str(e)}")
#             return jsonify({'error': 'Error during prediction'}), 400  # More generic error message

#         # Preprocess the audio to match the model's expected input shape
#         features = preprocess_audio(audio_segment)

#         if interpreter is None:
#             return jsonify({'error': 'Model not loaded, unable to predict genre'}), 500

#         # Predict the genre using the TensorFlow Lite model
#         predicted_genre = predict_genre_with_tflite(features)

#         if predicted_genre is None:
#             return jsonify({'error': 'Error during prediction'}), 500

#         return jsonify({'predicted_genre': predicted_genre})

#     except Exception as e:
#         logging.error(f"Error in classification: {str(e)}")
#         return jsonify({'error': 'Error during prediction'}), 500  # Catch-all for unexpected errors


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002)
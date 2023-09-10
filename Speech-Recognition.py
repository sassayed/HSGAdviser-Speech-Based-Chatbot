import speech_recognition as sr

recognizer = sr.Recognizer()

microphone = sr.Microphone()

with microphone as source:
    print("Say your question please?")
    audio_data = recognizer.listen(source)
    print("Audio recorded completed.")

try:
    speechtext = recognizer.recognize_google(audio_data)
    print("Recognized text:", speechtext)
except sr.UnknownValueError:
    print("Speech recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")

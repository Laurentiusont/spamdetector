from django.shortcuts import render
from django.http import JsonResponse
import pickle
import os
from .train_model import preprocess_pipeline
# from apps.detector import train_model

# from apps.detector import train_model
# import train_model.EmailToWordCounterTransformer()
# import train_model.WordCounterToVectorTransformer()


def index(request):
    return render(request, 'detector/index.html')


def predict(request):
    if request.method == 'POST':
        message = [request.POST.get('message')]

        # Load the saved model and vectorizer
        model_path = os.path.join(os.path.dirname(
            __file__), 'spam_detector_model.pkl')
        # vectorizer_path = os.path.join(
        #     os.path.dirname(__file__), 'preprocess_pipeline.pkl')
        with open(model_path, 'rb') as model_file:
            clf = pickle.load(model_file)
        # with open(vectorizer_path, 'rb') as vectorizer_file:
        #     count_vectorizer = pickle.load(vectorizer_file)

        # Make a prediction
        message_vector = preprocess_pipeline.transform(message)
        prediction = clf.predict(message_vector)
        if (prediction[0] == 0):
            result = "Ham"
        else:
            result = "Spam"

        return JsonResponse({'prediction': result})
    else:
        return JsonResponse({'error': 'Invalid request method'})

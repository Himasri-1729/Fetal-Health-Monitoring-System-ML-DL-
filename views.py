from django.shortcuts import render
from .ml_model.predict_ml import predict_ml
import os
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load DL model once
dl_model_path = os.path.join(settings.BASE_DIR, 'dl_model', 'fetal_health_model.h5')
dl_model = tf.keras.models.load_model(dl_model_path)

# Class mapping from model output index
DL_CLASSES = ['Mild_Risk', 'Normal', 'Severe_Risk']

def home(request):
    ml_result = None
    dl_result = None
    ml_suggestions = None
    active_section = 'home'  # Default section

    if request.method == 'POST':
        form_type = request.POST.get('form_type')

        # 👉 ML Prediction Form
        if form_type == 'ml':
            active_section = 'ml'
            ml_input = {
                'baseline value': float(request.POST.get('baseline_value')),
                'accelerations': float(request.POST.get('accelerations')),
                'fetal_movement': float(request.POST.get('fetal_movement')),
                'uterine_contractions': float(request.POST.get('uterine_contractions')),
                'light_decelerations': float(request.POST.get('light_decelerations')),
                'severe_decelerations': float(request.POST.get('severe_decelerations')),
                'prolongued_decelerations': float(request.POST.get('prolongued_decelerations')),
                'abnormal_short_term_variability': float(request.POST.get('abnormal_short_term_variability')),
                'mean_value_of_short_term_variability': float(request.POST.get('mean_value_of_short_term_variability')),
                'percentage_of_time_with_abnormal_long_term_variability': float(request.POST.get('percentage_of_time_with_abnormal_long_term_variability')),
                'mean_value_of_long_term_variability': float(request.POST.get('mean_value_of_long_term_variability')),
                'histogram_width': float(request.POST.get('histogram_width')),
                'histogram_min': float(request.POST.get('histogram_min')),
                'histogram_max': float(request.POST.get('histogram_max')),
                'histogram_number_of_peaks': float(request.POST.get('histogram_number_of_peaks')),
                'histogram_number_of_zeroes': float(request.POST.get('histogram_number_of_zeroes')),
                'histogram_mode': float(request.POST.get('histogram_mode')),
                'histogram_mean': float(request.POST.get('histogram_mean')),
                'histogram_median': float(request.POST.get('histogram_median')),
                'histogram_variance': float(request.POST.get('histogram_variance')),
                'histogram_tendency': float(request.POST.get('histogram_tendency')),
            }

            ml_raw = predict_ml(ml_input)
            ml_result = {
                1: 'Normal',
                2: 'Mild Risk',
                3: 'Severe Risk'
            }.get(ml_raw, f"Unknown ({ml_raw})")
            # ML suggestions based on prediction
        ml_suggestions = None

        if ml_result == 'Normal':
            ml_suggestions = [
                "Continue regular check-ups with your doctor.",
                "Maintain a balanced diet and stay hydrated.",
                "Engage in light physical activity or prenatal yoga.",
                "Avoid unnecessary stress and get adequate rest."
            ]
        elif ml_result == 'Mild Risk':
            ml_suggestions = [
                "Monitor fetal movements regularly.",
                "Avoid strenuous physical activity.",
                "Consult your doctor for early interventions.",
                "Rest as much as possible and reduce workload."
            ]
        elif ml_result == 'Severe Risk':
            ml_suggestions = [
                "Seek immediate medical consultation.",
                "You may require close monitoring or hospitalization.",
                "Avoid all physical strain and stay in a calm environment.",
                "Follow your doctor's instructions strictly and avoid travel."
            ]


        # 👉 DL Image Prediction Form
        elif form_type == 'dl' and 'fetal_image' in request.FILES:
            active_section = 'dl'
            img_file = request.FILES['fetal_image']
            img_path = os.path.join(settings.MEDIA_ROOT, img_file.name)

            # Save uploaded image
            with open(img_path, 'wb+') as f:
                for chunk in img_file.chunks():
                    f.write(chunk)

            # Preprocess image for DL model
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            dl_pred = dl_model.predict(img_array)
            dl_index = np.argmax(dl_pred)
            dl_result = DL_CLASSES[dl_index]

    return render(request, 'home.html', {
    'ml_result': ml_result,
    'dl_result': dl_result,
    'ml_suggestions': ml_suggestions,
    'active_section': active_section,})


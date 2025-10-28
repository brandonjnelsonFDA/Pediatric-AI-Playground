from pathlib import Path
import os

import pandas as pd
import numpy as np
from model_utils import download_and_unzip, InferenceManager
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import nibabel as nib
import gradio as gr

load_dotenv()

hssayeni_dir = Path(os.environ['HSSAYENI_DIR'])
metadata = pd.read_csv(hssayeni_dir / 'Patient_demographics.csv')
names = []
ages = []
files = []
for _, row in metadata.iterrows():
    series_id = row['Patient Number']
    age = row['Age\n(years)']

    if pd.isna(series_id) or pd.isna(age):
        continue

    file_path = hssayeni_dir / 'ct_scans' / f'{int(series_id):03d}.nii'
    if not file_path.exists():
        continue

    names.append(int(series_id))
    ages.append(float(age))
    files.append(file_path)
patients = pd.DataFrame(dict(name=names, age=ages, file=files))

model_path = Path(os.environ.get('MODEL_PATH', '/scratch/brandon.nelson/demos/pediatric_ich_cadt/model_files'))
if not model_path.exists():
    download_and_unzip('https://zenodo.org/records/15750437/files/model_files.zip', extract_to=model_path)

models = {m.parts[-2]: InferenceManager(m) for m in sorted(list(model_path.rglob('*.pth')))}

display_settings = {
    'brain': (80, 40),
    'subdural': (300, 100),
    'stroke': (40, 40),
    'temporal bones': (2800, 600),
    'soft tissues': (400, 50),
}


def get_patient_number_from_dropdown(patient_string):
    if not patient_string:
        return None
    return int(patient_string.split(' ')[1])


def get_patient_images(patient_name):
    patient_number = get_patient_number_from_dropdown(patient_name)
    if patient_number is None:
        return None, None
    patient = patients[patients['name'] == patient_number]
    if not patient.empty:
        images = nib.load(patient['file'].iloc[0]).get_fdata().transpose(2, 1, 0)[:, ::-1]
        return images, patient['age'].iloc[0]
    return None, None


def load_patient_data(patient_name, model_name):
    images, _ = get_patient_images(patient_name)
    if images is not None:
        models[model_name].load_patient(images)
        max_slices = len(images) - 1
        return gr.update(maximum=max_slices, value=max_slices // 2)
    return gr.update(maximum=0, value=0)


def normalize(img, vmin=None, vmax=None):
    if vmin is not None:
        img = np.clip(img, a_min=vmin, a_max=vmax)
    return (img - img.min()) / (img.max() - img.min())

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

def visualize_ict_pipeline(patient_name, slice_num, width=5, thresh=0.3, model_name='CAD_1', avg_predictions=True, display_setting='brain'):
    if not patient_name:
        return None, None, "<p style='color:black'>Please select a patient.</p>"

    images, age = get_patient_images(patient_name)
    if images is None:
        return None, None, "<p style='color:black'>Patient not found.</p>"

    slice_num = int(slice_num) if slice_num is not None else len(images) // 2

    if models[model_name].patient_images is None:
        models[model_name].load_patient(images)

    if avg_predictions:
        slice_range = range(slice_num, slice_num + width)
        predictions = [models[model_name].get_slice_prediction(i) for i in slice_range]
        out = {k: np.mean([p[k] for p in predictions if p is not None]) for k in models[model_name].labels}
        image = np.mean(images[slice_num:slice_num + width], axis=0)
    else:
        image = np.mean(images[slice_num:slice_num + width], axis=0)
        out = models[model_name].predict_image_on_the_fly(image)

    diagnosis = pd.read_csv(hssayeni_dir / 'hemorrhage_diagnosis_raw_ct.csv')
    patient_number = get_patient_number_from_dropdown(patient_name)
    label_row = diagnosis.loc[(diagnosis.PatientNumber == patient_number) & (diagnosis.SliceNumber == slice_num + 1)]

    subtype = 'No_Hemorrhage'
    if not label_row.empty:
        label = label_row.to_numpy()[:, 2:-1]
        cols = diagnosis.columns[2:-1]
        if label.size > 0:
            subtype = cols[label.argmax()]

    out_copy = out.copy()
    out_copy.pop('Any', None)
    max_label = max(out_copy, key=out_copy.get) if out_copy else 'No_Hemorrhage'
    predicted_label = max_label if out_copy.get(max_label, 0) > thresh else 'No_Hemorrhage'

    color = "green" if predicted_label == subtype else "red"
    prediction_text = f"age: {age}, model prediction: {predicted_label} | truth: {subtype}"

    fig.suptitle(prediction_text, color=color)
    window, level = display_settings[display_setting]
    vmin = level - window // 2
    vmax = level + window //2
    axs[0].clear()
    axs[0].imshow(image, vmin=vmin, vmax=vmax, cmap='gray')
    axs[0].set_axis_off()

    keys = list(out.keys())
    values = list(out.values())
    bar_colors = []
    for key, value in out.items():
        if key == 'Any':
            # "Any" label logic
            is_hemorrhage_present = subtype != 'No_Hemorrhage'
            is_prediction_positive = value > thresh

            if is_prediction_positive and is_hemorrhage_present:
                bar_colors.append('green')  # True Positive
            elif not is_prediction_positive and is_hemorrhage_present:
                bar_colors.append('red')  # False Negative
            elif is_prediction_positive and not is_hemorrhage_present:
                bar_colors.append('red')  # False Positive
            else:
                bar_colors.append('blue')  # True Negative
        else:
            # Original logic for other subtypes
            # True Positive
            if value > thresh and key == subtype:
                bar_colors.append('green')
            # False Positive
            elif value > thresh and key != subtype:
                bar_colors.append('red')
            # False Negative
            elif value <= thresh and key == subtype:
                bar_colors.append('red')
            # True Negative
            else:
                bar_colors.append('blue')
    axs[1].clear()
    axs[1].bar(keys, values, color=bar_colors)
    axs[1].set_ylabel('model output')
    axs[1].set_ylim([0, 1])
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].hlines(thresh, 0, len(out) -1, colors='red')
    axs[1].text(len(out) - 1, thresh + 0.01, 'detection threshold', ha='right', va='bottom', fontsize='small', color='gray')
    plt.tight_layout()

    return fig


def update_patient_dropdown(min_age, max_age):
    filtered_patients = patients[(patients['age'] >= min_age) & (patients['age'] <= max_age)]
    choices = [f"Patient {name} - Age {age}" for name, age in zip(filtered_patients['name'], filtered_patients['age'])]
    return gr.update(choices=choices)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                min_age_input = gr.Number(label="Min Age", value=0)
                max_age_input = gr.Number(label="Max Age", value=99)
            patient_selector = gr.Dropdown(choices=[f"Patient {name} - Age {age}" for name, age in zip(patients['name'], patients['age'])], label="Patient Number")
            slice_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Slice Number")
            width_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Width")
            thresh_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label="Threshold")
            model_selector = gr.Dropdown(choices=list(models.keys()), label="Model Name", value='CAD_1')
            avg_predictions_checkbox = gr.Checkbox(label="Average predictions (faster)", value=True)
        with gr.Column(scale=2):
            image_output = gr.Plot(label="CT Slice")
            display_settings_selector = gr.Dropdown(choices=list(display_settings.keys()), label="Display Settings", value='brain')

    patient_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=slice_slider)
    model_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=slice_slider)
    min_age_input.change(fn=update_patient_dropdown, inputs=[min_age_input, max_age_input], outputs=patient_selector)
    max_age_input.change(fn=update_patient_dropdown, inputs=[min_age_input, max_age_input], outputs=patient_selector)

    inputs = [patient_selector, slice_slider, width_slider, thresh_slider, model_selector, avg_predictions_checkbox, display_settings_selector]
    outputs = [image_output]

    patient_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    slice_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    width_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    thresh_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    model_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    avg_predictions_checkbox.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    display_settings_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()

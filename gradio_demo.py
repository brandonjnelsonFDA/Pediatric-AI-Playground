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
for series_id in metadata['Patient Number']:
    if np.isnan(series_id):
        continue
    names.append(int(series_id))
    ages.append(float(metadata[metadata['Patient Number'] == series_id]['Age\n(years)']))
    files.append(hssayeni_dir / 'ct_scans' / f'{int(series_id):03d}.nii')
patients = pd.DataFrame(dict(name=names, age=ages, file=files))

model_path = Path(os.environ.get('MODEL_PATH', '/scratch/brandon.nelson/demos/pediatric_ich_cadt/model_files'))
if not model_path.exists():
    download_and_unzip('https://zenodo.org/records/15750437/files/model_files.zip', extract_to=model_path)

models = {m.parts[-2]: InferenceManager(m) for m in sorted(list(model_path.rglob('*.pth')))}


def get_patient_images(patient_name):
    patient = patients[patients['name'] == int(patient_name)]
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


def visualize_ict_pipeline(patient_name, slice_num, width=5, thresh=0.3, model_name='CAD_1', avg_predictions=True):
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
    label_row = diagnosis.loc[(diagnosis.PatientNumber == int(patient_name)) & (diagnosis.SliceNumber == slice_num + 1)]

    subtype = 'Normal'
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
    prediction_text = f"<p style='color:{color}'>age: {age}, <br>model prediction: {predicted_label} | truth: {subtype}</p>"

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(out.keys(), out.values())
    ax.set_ylabel('model output')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', labelrotation=45)
    ax.hlines(thresh, 0, len(out) -1, colors='red')
    plt.tight_layout()

    window = 300
    level = 150
    vmin = level - window // 2
    vmax = level + window //2
    image = normalize(image, vmin=vmin, vmax=vmax)
    return image, fig, prediction_text

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            patient_selector = gr.Dropdown(choices=[str(name) for name in patients['name']], label="Patient Number")
            slice_slider = gr.Slider(minimum=0, maximum=100, step=1, label="Slice Number")
            width_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Width")
            thresh_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label="Threshold")
            model_selector = gr.Dropdown(choices=list(models.keys()), label="Model Name", value='CAD_1')
            avg_predictions_checkbox = gr.Checkbox(label="Average predictions (faster)", value=True)
        with gr.Column(scale=2):
            image_output = gr.Image(label="CT Slice")
            prediction_label = gr.HTML(label="Prediction")
            plot_output = gr.Plot(label="Model Output")

    patient_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=slice_slider)
    model_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=slice_slider)

    inputs = [patient_selector, slice_slider, width_slider, thresh_slider, model_selector, avg_predictions_checkbox]
    outputs = [image_output, plot_output, prediction_label]

    patient_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    slice_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    width_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    thresh_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    model_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    avg_predictions_checkbox.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()

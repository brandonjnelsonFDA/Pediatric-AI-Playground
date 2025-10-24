from pathlib import Path
import pandas as pd
import numpy as np
from model_utils import download_and_unzip
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from model_utils import predict_image

import albumentations as A
import torch
import nibabel as nib
import gradio as gr

load_dotenv()


hssayeni_dir = Path('/projects01/didsr-aiml/brandon.nelson/pedsilicoICH/datasets/Hssayeni/')
metadata = pd.read_csv(hssayeni_dir / 'Patient_demographics.csv')
names = []
ages = []
files = []
for series_id in metadata['Patient Number']:
    if np.isnan(series_id):
        continue
    names.append(int(series_id))
    ages.append(float(metadata[metadata['Patient Number'] == series_id]['Age\n(years)']))
    files = hssayeni_dir / 'ct_scans' / f'{int(series_id):03d}.nii'
patients = pd.DataFrame(dict(name=names, age=ages, file=files))
patients

# model_path = Path(os.environ['MODEL_PATH'])
model_path = Path('/scratch/brandon.nelson/demos/pediatric_ich_cadt/model_files')
if not model_path.exists():
    download_and_unzip('https://zenodo.org/records/15750437/files/model_files.zip', extract_to=model_path.parents[1]) 

# %%
models = {m.parts[-2]: m for m in sorted(list(model_path.rglob('*.pth')))}

patient = patients.iloc[0]
images = nib.load(patient['file']).get_fdata().transpose(2, 1, 0)[:, ::-1]
name = patient['name']
age = patient['age']

def ict_pipeline(slice_num, width=5, model_name='CAD_1'):

    image = np.mean(images[slice_num:slice_num+width], axis=0) # create average
    out = predict_image(image, models[model_name], device='cuda')
    return image, out



def visualize_ict_pipeline(slice_num, width=5, thresh=0.3, model_name='CAD_1', show=True):
    diagnosis = pd.read_csv(hssayeni_dir / 'hemorrhage_diagnosis_raw_ct.csv')
    label = diagnosis.loc[(diagnosis.PatientNumber == name) & (diagnosis.SliceNumber == slice_num + 1)].to_numpy()[:, 2:-1]
    cols = diagnosis.columns[2:-1]
    subtype = cols[label.argmax()]

    image, out = ict_pipeline(slice_num, width, model_name)

    f, axs = plt.subplots(1, 2, figsize = (10, 4), dpi=150)
    axs[0].imshow(image, vmin=0, vmax=80, cmap='gray')
    axs[0].set_axis_off()
    axs[1].bar(out.keys(), out.values())
    axs[1].set_ylabel('model output')
    axs[1].set_ylim([0, 1])
    axs[1].hlines(thresh, 0, len(out), colors='red')
    out.pop('Any')
    max_label = [k for k, v in out.items() if v == max(out.values())][0]
    predicted_label = max_label if out[max_label] > thresh else 'No_Hemorrhage'
    color = 'green' if predicted_label == subtype else 'red'
    axs[0].set_title(f'age: {age}, \nmodel prediction: {predicted_label} | truth: {subtype}', color=color)
    fname = Path('results.png').absolute()
    if show:
        plt.show()
        return None
    else:
        plt.savefig(fname)
        return fname

# Create the Gradio interface
iface = gr.Interface(
    fn=visualize_ict_pipeline,
    inputs=[
        gr.Slider(minimum=0, maximum=len(images), step=1, label="Slice Number"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Width"),
        gr.Slider(minimum=0, maximum=1, step=0.1, label="Threshold"),
        gr.Dropdown(choices=list(models.keys()), label="Model Name")
    ],
    outputs=["image"]
)

# Launch the interface
iface.launch()

# %%





from pathlib import Path
import os

import pandas as pd
import numpy as np
from model_utils import download_and_unzip, InferenceManager
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import nibabel as nib
import gradio as gr
import SimpleITK as sitk

load_dotenv()


def load_hssayeni_data(hssayeni_dir):
    hssayeni_dir = Path(hssayeni_dir)
    metadata = pd.read_csv(hssayeni_dir / 'Patient_demographics.csv')
    # Rename the problematic column
    metadata.rename(columns={'Age\n(years)': 'age_years'}, inplace=True)
    diagnosis = pd.read_csv(hssayeni_dir / 'hemorrhage_diagnosis_raw_ct.csv')
    rows = []
    for idx, row in diagnosis.iterrows():
        series_id = row.PatientNumber
        if np.isnan(series_id):
            continue
        row['name'] = int(series_id)
        row['age'] = float(metadata[metadata['Patient Number'] == series_id]['age_years'].iloc[0])
        row['file'] = hssayeni_dir / 'ct_scans' / f'{int(series_id):03d}.nii'
        rows.append(row)
    return pd.DataFrame(rows)


synth_labels_to_real = {
    'EDH': 'Epidural',
    'SDH': 'Subdural',
    'IPH': 'Intraparenchymal',
    'IVH': 'Intraventricular',
    'SAH': 'Subarachnoid',
    None: 'No_Hemorrhage',
}


def load_synthetic_data(synth_dir):
    synth_dir = Path(synth_dir)
    results = pd.concat([pd.read_csv(o) for o in synth_dir.rglob('*.csv')])
    results.loc[results['lesion_volume(mL)'] == 0, 'subtype'] = None
    diagnosis = pd.get_dummies(results.subtype.apply(lambda o: synth_labels_to_real.get(o, 'No_Hemorrhage'))).astype(float)
    results = pd.concat([results, diagnosis], axis=1)
    results['file'] = results.case_id.apply(lambda o: synth_dir / o / 'dicoms')
    results['age'] = results.phantom.apply(lambda o: float(o.split(' yr')[0]))
    results['name'] = results.case_id.apply(lambda o: f"synthetic {int(o.split('_')[-1])}")
    results['SliceNumber'] = results['image_file_path'].apply(lambda o: int(Path(o).stem.split('_')[-1]))
    return results


def load_datasets(hssayeni_dir=None, synth_dir=None):
    hssayeni_patients = load_hssayeni_data(hssayeni_dir) if hssayeni_dir else pd.DataFrame()
    
    if synth_dir:
        synth_dir = Path(synth_dir)
        # Check if data exists
        if not list(synth_dir.rglob('*.csv')):
            print("Synthetic data not found. Downloading...")
            download_and_unzip('https://zenodo.org/records/15691337/files/manuscript_100_280mA_wME.zip', extract_to=synth_dir)
    
    synth_patients = load_synthetic_data(synth_dir) if synth_dir else pd.DataFrame()

    patients_list = []
    if not hssayeni_patients.empty:
        hssayeni_patients['dataset'] = 'Hssayeni'
        patients_list.append(hssayeni_patients)
    if not synth_patients.empty:
        synth_patients['dataset'] = 'Synthetic'
        patients_list.append(synth_patients)

    if not patients_list:
        return pd.DataFrame(columns=['name', 'age', 'dataset', *synth_labels_to_real.values(), 'file'])

    patients = pd.concat(patients_list, ignore_index=True)

    # Ensure all required columns are present.
    required_cols = ['name', 'age', 'dataset', 'SliceNumber', *synth_labels_to_real.values(), 'file']
    for col in required_cols:
        if col not in patients.columns:
            patients[col] = 0.0

    patients = patients[required_cols]
    patients.fillna(0.0, inplace=True)
    return patients

hssayeni_dir = Path(os.environ['HSSAYENI_DIR'])
synth_dir = Path(os.environ['SYNTH_DIR'])
patients = load_datasets(hssayeni_dir, synth_dir)

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


def get_patient_name_from_dropdown(patient_string):
    if not patient_string:
        return None
    parts = patient_string.split(' ')
    if parts[1] == 'synthetic':
        return f"synthetic {parts[2]}"
    else:
        return int(float(parts[1]))


def get_patient_images(patient_name):
    patient_identifier = get_patient_name_from_dropdown(patient_name)
    if patient_identifier is None:
        return None, None
    patient = patients[patients['name'] == patient_identifier]
    if not patient.empty:
        filepath = Path(patient['file'].iloc[0])
        if filepath.is_dir(): # DICOM
            images = sitk.GetArrayFromImage(sitk.ReadImage(sorted(list(filepath.glob('*.dcm')))))
        else: # NIfTI
            images = nib.load(filepath).get_fdata().transpose(2, 1, 0)[:, ::-1]
        return images, patient['age'].iloc[0]
    return None, None


def get_hemorrhage_info(patient_df):
    """
    Analyzes patient data to find hemorrhage slices and format information.

    Args:
        patient_df (pd.DataFrame): DataFrame with slice-wise data for a single patient.

    Returns:
        tuple: A tuple containing:
            - int: The ideal starting slice number.
            - str: A formatted string describing hemorrhage locations.
    """
    hemorrhage_types = ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid']
    hemorrhage_info = {}

    for hemorrhage_type in hemorrhage_types:
        if hemorrhage_type not in patient_df.columns or patient_df[hemorrhage_type].sum() == 0:
            continue

        slices = patient_df[patient_df[hemorrhage_type] == 1]['SliceNumber'].tolist()

        if not slices:
            continue

        slices.sort()
        ranges = []
        start_slice = slices[0]
        for i in range(1, len(slices)):
            if slices[i] != slices[i-1] + 1:
                ranges.append((start_slice, slices[i-1]))
                start_slice = slices[i]
        ranges.append((start_slice, slices[-1]))

        if ranges:
            hemorrhage_info[hemorrhage_type] = ranges

    all_ranges_with_type = []
    for hemo_type, ranges in hemorrhage_info.items():
        for r in ranges:
            all_ranges_with_type.append({'type': hemo_type, 'start': r[0], 'end': r[1]})

    all_ranges_with_type.sort(key=lambda x: x['start'])

    info_str_parts = []
    for r_info in all_ranges_with_type:
        hemo_type = r_info['type']
        start, end = r_info['start'], r_info['end']
        if start == end:
            info_str_parts.append(f"{hemo_type}: {start}")
        else:
            info_str_parts.append(f"{hemo_type}: {start}-{end}")

    info_str = ", ".join(info_str_parts)

    initial_slice = len(patient_df) // 2
    if all_ranges_with_type:
        first_hemorrhage = all_ranges_with_type[0]
        initial_slice = (first_hemorrhage['start'] + first_hemorrhage['end']) // 2

    return initial_slice, info_str


def load_patient_data(patient_name, model_name):
    images, _ = get_patient_images(patient_name)
    if images is not None:
        models[model_name].load_patient(images)
        max_slices = len(images) - 1

        patient_identifier = get_patient_name_from_dropdown(patient_name)
        patient_df = patients[patients['name'] == patient_identifier]

        initial_slice, info_str = get_hemorrhage_info(patient_df)

        return gr.update(maximum=max_slices, value=initial_slice), info_str
    return gr.update(maximum=0, value=0), ""


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

    patient_identifier = get_patient_name_from_dropdown(patient_name)
    label_row = patients.loc[(patients.name == patient_identifier) & (patients.SliceNumber == slice_num + 1)]

    truth_labels = set()
    if not label_row.empty:
        label_vector = label_row.iloc[:, 4:-1].to_numpy()[0]
        cols = patients.columns[4:-1]
        truth_labels = {cols[i] for i, val in enumerate(label_vector) if val == 1}

    if not truth_labels:
        truth_labels.add('No_Hemorrhage')

    title_parts = []
    title_color = 'green'
    hemorrhage_types = [col for col in patients.columns[4:-1] if col != 'No_Hemorrhage' and col in out]

    all_true_negatives = True
    for hemorrhage_type in hemorrhage_types:
        is_present_in_truth = hemorrhage_type in truth_labels
        is_predicted = out.get(hemorrhage_type, 0) > thresh

        if is_present_in_truth and is_predicted:  # True Positive
            title_parts.append(f'{hemorrhage_type} (TP)')
            all_true_negatives = False
        elif not is_present_in_truth and is_predicted:  # False Positive
            title_parts.append(f'{hemorrhage_type} (FP)')
            title_color = 'red'
            all_true_negatives = False
        elif is_present_in_truth and not is_predicted:  # False Negative
            title_parts.append(f'{hemorrhage_type} (FN)')
            title_color = 'red'
            all_true_negatives = False
        # True negatives are ignored in the title

    if all_true_negatives:
        prediction_text = "No hemorrhages detected"
        title_color = 'black'
    else:
        truth_text = f"Truth: {', '.join(sorted(list(truth_labels)))}"
        prediction_text = f"Predictions: {', '.join(title_parts)}"
        prediction_text = f"Age: {age} | {truth_text} | {prediction_text}"

    # Matplotlib's suptitle does not support multi-colored text.
    # As a workaround, we set the entire title to red if any prediction is incorrect.
    fig.suptitle(prediction_text, color=title_color)
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
        is_present_in_truth = key in truth_labels
        is_predicted = value > thresh

        if is_present_in_truth and is_predicted:  # True Positive
            bar_colors.append('green')
        elif not is_present_in_truth and is_predicted:  # False Positive
            bar_colors.append('red')
        elif is_present_in_truth and not is_predicted:  # False Negative
            bar_colors.append('red')
        else:  # True Negative
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


def update_patient_dropdown(min_age, max_age, datasets):
    filtered_patients = patients[
        (patients['age'] >= min_age) &
        (patients['age'] <= max_age) &
        (patients['dataset'].isin(datasets))
    ]
    choices = list(set([f"Patient {name} - Age {age}" for name, age in zip(filtered_patients['name'], filtered_patients['age'])]))
    choices.sort(key=lambda o: float(o.split(' - Age')[0].split(' ')[-1]))
    return gr.update(choices=choices)


patient_list = list(set([f"Patient {name} - Age {age}" for name, age in zip(patients['name'], patients['age'])]))
patient_list.sort(key=lambda o: float(o.split(' - Age')[0].split(' ')[-1]))


default_patient_string = None
default_patient_max_slice = 100
initial_slice = default_patient_max_slice // 2
info_str = ''
if not patients.empty:
    # Find patient 77 from Hssayeni dataset
    patient_77 = patients[(patients['name'] == 77) & (patients['dataset'] == 'Hssayeni')]
    default_patient_max_slice = patient_77.SliceNumber.max()
    initial_slice, info_str = get_hemorrhage_info(patient_77)

    if not patient_77.empty:
        age_77 = patient_77['age'].iloc[0]
        potential_default = f"Patient {77.0} - Age {age_77}"
        # Make sure it's in the list
        if potential_default in patient_list:
            default_patient_string = potential_default

    if default_patient_string is None and patient_list:
        # Fallback to the first patient
        default_patient_string = patient_list[0]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                min_age_input = gr.Number(label="Min Age", value=0)
                max_age_input = gr.Number(label="Max Age", value=99)
            dataset_selector = gr.CheckboxGroup(choices=['Hssayeni', 'Synthetic'], label='Datasets', value=['Hssayeni', 'Synthetic'])
            patient_selector = gr.Dropdown(choices=patient_list, label="Patient Number", value=default_patient_string)
            slice_slider = gr.Slider(minimum=0, maximum=default_patient_max_slice, step=1, label="Slice Number", value=initial_slice)
            width_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Width")
            thresh_slider = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.3, label="Threshold")
            model_selector = gr.Dropdown(choices=list(models.keys()), label="Model Name", value='CAD_1')
            avg_predictions_checkbox = gr.Checkbox(label="Average predictions (faster)", value=True)
        with gr.Column(scale=2):
            image_output = gr.Plot(label="CT Slice")
            display_settings_selector = gr.Dropdown(choices=list(display_settings.keys()), label="Display Settings", value='brain')
            hemorrhage_info_box = gr.Textbox(label="Hemorrhage Info", interactive=False, value=info_str)

    patient_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=[slice_slider, hemorrhage_info_box])
    model_selector.change(fn=load_patient_data, inputs=[patient_selector, model_selector], outputs=[slice_slider, hemorrhage_info_box])
    min_age_input.change(fn=update_patient_dropdown, inputs=[min_age_input, max_age_input, dataset_selector], outputs=patient_selector)
    max_age_input.change(fn=update_patient_dropdown, inputs=[min_age_input, max_age_input, dataset_selector], outputs=patient_selector)
    dataset_selector.change(fn=update_patient_dropdown, inputs=[min_age_input, max_age_input, dataset_selector], outputs=patient_selector)

    inputs = [patient_selector, slice_slider, width_slider, thresh_slider, model_selector, avg_predictions_checkbox, display_settings_selector]
    outputs = [image_output]

    patient_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    slice_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    width_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    thresh_slider.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    model_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    avg_predictions_checkbox.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)
    display_settings_selector.change(fn=visualize_ict_pipeline, inputs=inputs, outputs=outputs)

    demo.load(
        fn=visualize_ict_pipeline,
        inputs=[
            patient_selector,
            slice_slider,
            width_slider,
            thresh_slider,
            model_selector,
            avg_predictions_checkbox,
            display_settings_selector
        ],
        outputs=outputs
    )

if __name__ == "__main__":
    demo.launch()

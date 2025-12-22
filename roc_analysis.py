import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from gradio_demo
from gradio_demo import (
    load_datasets,
    models,
    get_patient_images,
    get_patient_name_from_dropdown,
    hssayeni_dir,
    synth_dir
)

def get_ground_truth_and_predictions(patients_df, dataset_name):
    """
    Runs inference for all patients in the specified dataset and aligns
    predictions with ground truth. Returns separate results for slice-level
    and patient-level aggregation.
    """
    print(f"Processing dataset: {dataset_name}")
    dataset_df = patients_df[patients_df['dataset'] == dataset_name]

    results = {
        'CAD_1': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}},
        'CAD_2': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}},
        'CAD_3': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}}
    }

    subtypes = ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid']

    unique_patients = dataset_df['name'].unique()

    for patient_name in tqdm(unique_patients, desc=f"Evaluating {dataset_name}"):
        patient_row = dataset_df[dataset_df['name'] == patient_name].iloc[0]
        age = patient_row['age']
        patient_ui_string = f"Patient {patient_name} - Age {age}"

        images, _ = get_patient_images(patient_ui_string)

        if images is None:
            print(f"Warning: Could not load images for {patient_ui_string}")
            continue

        patient_gt = dataset_df[dataset_df['name'] == patient_name]

        # Determine Patient-Level Ground Truth
        # Patient is positive for a subtype if ANY slice has that subtype
        patient_gt_any_slice = {}
        for subtype in subtypes:
            patient_gt_any_slice[subtype] = 1.0 if patient_gt[subtype].sum() > 0 else 0.0
        patient_gt_any_slice['Any'] = 1.0 if any(patient_gt_any_slice[st] == 1.0 for st in subtypes) else 0.0

        for model_name, model in models.items():
            model.load_patient(images)

            patient_max_probs = {st: 0.0 for st in subtypes + ['Any']}

            for _, row in patient_gt.iterrows():
                slice_num = int(row['SliceNumber'])
                slice_idx = slice_num - 1

                if slice_idx >= len(images):
                    continue

                prediction = model.get_slice_prediction(slice_idx)
                if prediction is None:
                    continue

                # --- Slice Level ---
                gt_row = {}
                for subtype in subtypes:
                    gt_row[subtype] = row[subtype]
                gt_row['Any'] = 1.0 if any(row[subtype] == 1.0 for subtype in subtypes) else 0.0

                results[model_name]['Slice']['y_true'].append(gt_row)
                results[model_name]['Slice']['y_score'].append(prediction)

                # --- Patient Level Aggregation ---
                for key in patient_max_probs:
                    if key in prediction:
                        patient_max_probs[key] = max(patient_max_probs[key], prediction[key])

            # Store Patient Level Results
            results[model_name]['Patient']['y_true'].append(patient_gt_any_slice)
            results[model_name]['Patient']['y_score'].append(patient_max_probs)

    return results

def plot_roc_curves(results, dataset_name, output_dir):
    """
    Generates and saves ROC curves for the given results.
    Generates distinct plots for Slice-level and Patient-level.
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Any']
    auc_scores = []

    for level in ['Slice', 'Patient']:

        # 1. Overall Hemorrhage Detection (Any)
        plt.figure(figsize=(8, 6))
        plotted_any = False
        for model_name, model_data in results.items():
            data = model_data[level]
            if not data['y_true']:
                continue

            y_true = [item['Any'] for item in data['y_true']]
            y_score = [item['Any'] for item in data['y_score']]

            # Skip if only one class present
            if len(set(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            auc_scores.append({'Dataset': dataset_name, 'Level': level, 'Model': model_name, 'Type': 'Any', 'AUC': roc_auc})

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            plotted_any = True

        if plotted_any:
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{level}-Level Overall Hemorrhage Detection - {dataset_name}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, f'{dataset_name}_{level}_Overall_ROC.png'))
            plt.close()

        # 2. Per Subtype
        for subtype in classes[:-1]:
            plt.figure(figsize=(8, 6))
            plotted_subtype = False
            for model_name, model_data in results.items():
                data = model_data[level]
                if not data['y_true']:
                    continue

                y_true = [item[subtype] for item in data['y_true']]
                y_score = [item[subtype] for item in data['y_score']]

                if len(set(y_true)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                auc_scores.append({'Dataset': dataset_name, 'Level': level, 'Model': model_name, 'Type': subtype, 'AUC': roc_auc})

                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                plotted_subtype = True

            if plotted_subtype:
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{level}-Level {subtype} Detection - {dataset_name}')
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'{dataset_name}_{level}_{subtype}_ROC.png'))
                plt.close()

    return pd.DataFrame(auc_scores)

def main():
    print("Loading datasets...")
    patients_df = load_datasets(hssayeni_dir, synth_dir)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    all_auc_scores = []

    # Process Hssayeni
    if not patients_df[patients_df['dataset'] == 'Hssayeni'].empty:
        results_hssayeni = get_ground_truth_and_predictions(patients_df, 'Hssayeni')
        auc_hssayeni = plot_roc_curves(results_hssayeni, 'Hssayeni', output_dir)
        all_auc_scores.append(auc_hssayeni)

    # Process Synthetic
    if not patients_df[patients_df['dataset'] == 'Synthetic'].empty:
        results_synth = get_ground_truth_and_predictions(patients_df, 'Synthetic')
        auc_synth = plot_roc_curves(results_synth, 'Synthetic', output_dir)
        all_auc_scores.append(auc_synth)

    # Save AUC Report
    if all_auc_scores:
        final_auc_df = pd.concat(all_auc_scores, ignore_index=True)
        final_auc_df.to_csv(os.path.join(output_dir, 'auc_scores.csv'), index=False)
        print("\nAUC Scores:")
        print(final_auc_df)
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

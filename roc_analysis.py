import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import warnings
import random

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

def get_ground_truth_and_predictions_with_metadata(patients_df, dataset_name):
    """
    Runs inference for all patients in the specified dataset.
    Returns:
    1. results: Dictionary of aligned ground truth and predictions for ROC calc.
    2. metadata: List of dicts containing image paths, predictions, and ground truth
       to facilitate montage generation.
    """
    print(f"Processing dataset: {dataset_name}")
    dataset_df = patients_df[patients_df['dataset'] == dataset_name]

    results = {
        'CAD_1': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}},
        'CAD_2': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}},
        'CAD_3': {'Slice': {'y_true': [], 'y_score': []}, 'Patient': {'y_true': [], 'y_score': []}}
    }

    # Metadata for montage generation
    metadata = []

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

        # Patient Level Ground Truth
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

                # Store metadata
                metadata.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'patient_ui_string': patient_ui_string,
                    'slice_idx': slice_idx,
                    'y_true': gt_row,
                    'y_score': prediction
                })

                # --- Patient Level Aggregation ---
                for key in patient_max_probs:
                    if key in prediction:
                        patient_max_probs[key] = max(patient_max_probs[key], prediction[key])

            results[model_name]['Patient']['y_true'].append(patient_gt_any_slice)
            results[model_name]['Patient']['y_score'].append(patient_max_probs)

    return results, metadata

def plot_roc_curves_with_thresholds(results, dataset_name, output_dir):
    """
    Generates ROC curves and calculates thresholds for specific sensitivities.
    Returns:
        auc_df: DataFrame of AUC scores.
        thresholds: Nested dict of thresholds for [Model][Level][Type][Sensitivity]
    """
    os.makedirs(output_dir, exist_ok=True)

    classes = ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Any']
    auc_scores = []

    # Store thresholds: thresholds[model][level][type][0.9] = val
    calculated_thresholds = {}

    for level in ['Slice', 'Patient']:

        # 1. Overall Hemorrhage Detection (Any)
        plt.figure(figsize=(10, 8))
        plotted_any = False

        for model_name, model_data in results.items():
            if model_name not in calculated_thresholds:
                calculated_thresholds[model_name] = {'Slice': {}, 'Patient': {}}

            data = model_data[level]
            if not data['y_true']:
                continue

            y_true = [item['Any'] for item in data['y_true']]
            y_score = [item['Any'] for item in data['y_score']]

            if len(set(y_true)) < 2:
                continue

            fpr, tpr, thres = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            auc_scores.append({'Dataset': dataset_name, 'Level': level, 'Model': model_name, 'Type': 'Any', 'AUC': roc_auc})

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            plotted_any = True

            # Calculate Thresholds
            target_sensitivities = [0.1, 0.5, 0.9]
            model_thresholds = {}

            colors = {0.1: 'red', 0.5: 'orange', 0.9: 'green'}

            for target_sens in target_sensitivities:
                idx = np.argmin(np.abs(tpr - target_sens))
                threshold_val = thres[idx]
                actual_sens = tpr[idx]
                actual_fpr = fpr[idx]

                model_thresholds[target_sens] = threshold_val

                plt.plot(actual_fpr, actual_sens, marker='o', markersize=5, color=colors[target_sens])
                if level == 'Slice' and model_name == 'CAD_1':
                     plt.text(actual_fpr, actual_sens, f'{target_sens*100:.0f}% Sens', fontsize=8)

            calculated_thresholds[model_name][level]['Any'] = model_thresholds

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
            plt.figure(figsize=(10, 8))
            plotted_subtype = False
            for model_name, model_data in results.items():
                data = model_data[level]
                if not data['y_true']:
                    continue

                y_true = [item[subtype] for item in data['y_true']]
                y_score = [item[subtype] for item in data['y_score']]

                if len(set(y_true)) < 2:
                    continue

                fpr, tpr, thres = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                auc_scores.append({'Dataset': dataset_name, 'Level': level, 'Model': model_name, 'Type': subtype, 'AUC': roc_auc})

                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                plotted_subtype = True

                # Calculate thresholds
                if subtype not in calculated_thresholds[model_name][level]:
                    calculated_thresholds[model_name][level][subtype] = {}

                target_sensitivities = [0.1, 0.5, 0.9]
                colors = {0.1: 'red', 0.5: 'orange', 0.9: 'green'}
                for target_sens in target_sensitivities:
                    idx = np.argmin(np.abs(tpr - target_sens))
                    threshold_val = thres[idx]
                    calculated_thresholds[model_name][level][subtype][target_sens] = threshold_val
                    plt.plot(fpr[idx], tpr[idx], marker='o', markersize=5, color=colors[target_sens])

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

    return pd.DataFrame(auc_scores), calculated_thresholds

def calculate_performance_metrics(results, thresholds, dataset_name):
    """
    Calculates PPV, NPV, Prevalence, and Counts at 90% Sensitivity.
    """
    metrics = []
    target_sens = 0.9

    classes = ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid', 'Any']

    for level in ['Slice', 'Patient']:
        for model_name, model_data in results.items():
            if model_name not in thresholds or level not in thresholds[model_name]:
                continue

            data = model_data[level]
            if not data['y_true']:
                continue

            for subtype in classes:
                if subtype not in thresholds[model_name][level]:
                    continue
                if target_sens not in thresholds[model_name][level][subtype]:
                    continue

                thresh_val = thresholds[model_name][level][subtype][target_sens]

                y_true = [item[subtype] for item in data['y_true']]
                y_score = [item[subtype] for item in data['y_score']]

                if not y_true:
                    continue

                # Convert to binary
                y_pred = [1.0 if score >= thresh_val else 0.0 for score in y_score]

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

                total = tn + fp + fn + tp
                prevalence = (tp + fn) / total if total > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                metrics.append({
                    'Dataset': dataset_name,
                    'Level': level,
                    'Model': model_name,
                    'Type': subtype,
                    'Threshold_90Sens': thresh_val,
                    'PPV': ppv,
                    'NPV': npv,
                    'Prevalence': prevalence,
                    'Count_Total': total,
                    'Count_Pos': tp + fn,
                    'Count_Neg': tn + fp,
                    'TP': tp,
                    'FP': fp,
                    'TN': tn,
                    'FN': fn
                })

    return pd.DataFrame(metrics)

def generate_montages(metadata, thresholds, output_dir, dataset_name):
    """
    Generates 3x3 montages for TP, TN, FP, FN at 90% Sensitivity.
    Applies windowing: Width 300, Level 150 -> vmin 0, vmax 300.
    """
    montage_dir = os.path.join(output_dir, 'Montages')
    os.makedirs(montage_dir, exist_ok=True)

    model_name = 'CAD_1'
    categories = ['Any'] + ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid']

    # Filter metadata for this model
    model_metadata = [m for m in metadata if m['model'] == model_name]
    if not model_metadata:
        return

    target_sens = 0.9
    sens_label = "HighSens" # Only High Sens requested

    # Thresholds for montage are primarily based on Slice level predictions
    # because metadata is slice-based.
    if model_name not in thresholds or 'Slice' not in thresholds[model_name]:
        return

    model_thresholds = thresholds[model_name]['Slice']

    for category in categories:
        if category not in model_thresholds or target_sens not in model_thresholds[category]:
            continue

        thresh_val = model_thresholds[category][target_sens]

        tp_list, tn_list, fp_list, fn_list = [], [], [], []

        for item in model_metadata:
            truth = item['y_true'][category]
            score = item['y_score'][category]
            pred = 1.0 if score >= thresh_val else 0.0

            if truth == 1.0 and pred == 1.0:
                tp_list.append(item)
            elif truth == 0.0 and pred == 0.0:
                tn_list.append(item)
            elif truth == 0.0 and pred == 1.0:
                fp_list.append(item)
            elif truth == 1.0 and pred == 0.0:
                fn_list.append(item)

        outcomes = {'TP': tp_list, 'TN': tn_list, 'FP': fp_list, 'FN': fn_list}

        for outcome_name, items_list in outcomes.items():
            if not items_list:
                continue

            n_samples = 9
            selected_items = random.sample(items_list, min(len(items_list), n_samples))

            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            fig.suptitle(f'{dataset_name} {model_name} {category} {outcome_name} ({sens_label}, Thresh={thresh_val:.2f})')

            for i, ax in enumerate(axes.flat):
                if i < len(selected_items):
                    item = selected_items[i]
                    images, _ = get_patient_images(item['patient_ui_string'])
                    if images is not None:
                        img_slice = images[item['slice_idx']]
                        # Apply Window: Level 150, Width 300 => vmin 0, vmax 300
                        ax.imshow(img_slice, cmap='gray', vmin=0, vmax=300)
                        ax.axis('off')
                        ax.set_title(f"Score: {item['y_score'][category]:.2f}")
                else:
                    ax.axis('off')

            plt.tight_layout()
            filename = f"{dataset_name}_{model_name}_{category}_{outcome_name}_{sens_label}.png"
            plt.savefig(os.path.join(montage_dir, filename))
            plt.close()

def plot_confusion_matrices(metadata, thresholds, output_dir, dataset_name):
    """
    Generates confusion matrices for 'Any' and subtypes at 90% sensitivity threshold.
    Using Slice level thresholds.
    """
    cm_dir = os.path.join(output_dir, 'Confusion_Matrices')
    os.makedirs(cm_dir, exist_ok=True)

    model_name = 'CAD_1'
    target_sens = 0.9

    model_metadata = [m for m in metadata if m['model'] == model_name]

    if model_name not in thresholds or 'Slice' not in thresholds[model_name]:
        return
    model_thresholds = thresholds[model_name]['Slice']

    categories = ['Any'] + ['Epidural', 'Subdural', 'Intraparenchymal', 'Intraventricular', 'Subarachnoid']

    for category in categories:
        if category not in model_thresholds or target_sens not in model_thresholds[category]:
            continue

        thresh_val = model_thresholds[category][target_sens]

        y_true = []
        y_pred = []

        for item in model_metadata:
            y_true.append(item['y_true'][category])
            pred_binary = 1.0 if item['y_score'][category] >= thresh_val else 0.0
            y_pred.append(pred_binary)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg', 'Pos'])

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'{dataset_name} {model_name} {category} CM (90% Sens)')

        filename = f"{dataset_name}_{model_name}_{category}_CM_90Sens.png"
        plt.savefig(os.path.join(cm_dir, filename))
        plt.close()

def main():
    print("Loading datasets...")
    patients_df = load_datasets(hssayeni_dir, synth_dir)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    all_auc_scores = []
    all_metrics = []

    # Process Hssayeni
    if not patients_df[patients_df['dataset'] == 'Hssayeni'].empty:
        results_hssayeni, meta_hssayeni = get_ground_truth_and_predictions_with_metadata(patients_df, 'Hssayeni')
        auc_hssayeni, thresholds_hssayeni = plot_roc_curves_with_thresholds(results_hssayeni, 'Hssayeni', output_dir)
        all_auc_scores.append(auc_hssayeni)

        metrics_hssayeni = calculate_performance_metrics(results_hssayeni, thresholds_hssayeni, 'Hssayeni')
        all_metrics.append(metrics_hssayeni)

        generate_montages(meta_hssayeni, thresholds_hssayeni, output_dir, 'Hssayeni')
        plot_confusion_matrices(meta_hssayeni, thresholds_hssayeni, output_dir, 'Hssayeni')

    # Process Synthetic
    if not patients_df[patients_df['dataset'] == 'Synthetic'].empty:
        results_synth, meta_synth = get_ground_truth_and_predictions_with_metadata(patients_df, 'Synthetic')
        auc_synth, thresholds_synth = plot_roc_curves_with_thresholds(results_synth, 'Synthetic', output_dir)
        all_auc_scores.append(auc_synth)

        metrics_synth = calculate_performance_metrics(results_synth, thresholds_synth, 'Synthetic')
        all_metrics.append(metrics_synth)

        generate_montages(meta_synth, thresholds_synth, output_dir, 'Synthetic')
        plot_confusion_matrices(meta_synth, thresholds_synth, output_dir, 'Synthetic')

    # Save AUC Report
    if all_auc_scores:
        final_auc_df = pd.concat(all_auc_scores, ignore_index=True)
        final_auc_df.to_csv(os.path.join(output_dir, 'auc_scores.csv'), index=False)
        print("\nAUC Scores:")
        print(final_auc_df)

    # Save Metrics Report
    if all_metrics:
        final_metrics_df = pd.concat(all_metrics, ignore_index=True)
        final_metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
        print("\nPerformance Metrics (90% Sensitivity):")
        print(final_metrics_df[['Dataset', 'Level', 'Model', 'Type', 'Threshold_90Sens', 'PPV', 'NPV', 'Prevalence']])

    else:
        print("No results generated.")

if __name__ == "__main__":
    main()

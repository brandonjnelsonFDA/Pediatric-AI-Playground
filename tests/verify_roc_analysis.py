import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import shutil
import sys

# Mock imports that might fail or are unnecessary for this test
sys.modules['gradio_demo'] = MagicMock()
sys.modules['model_utils'] = MagicMock()

class TestROCAnalysis(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = 'test_results'
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @patch('roc_analysis.load_datasets')
    @patch('roc_analysis.get_patient_images')
    @patch('roc_analysis.models')
    def test_end_to_end_roc_analysis(self, mock_models, mock_get_images, mock_load_datasets):
        # 1. Mock Data Loading - Realistic Subset
        data = {
            'name': [49, 49, 49, 54, 54, 54],
            'age': [35.0, 35.0, 35.0, 60.0, 60.0, 60.0],
            'dataset': ['Hssayeni']*6,
            'SliceNumber': [1, 2, 3, 1, 2, 3],
            'Epidural':         [0, 1, 0, 0, 0, 0],
            'Subdural':         [0, 0, 0, 0, 0, 0],
            'Intraparenchymal': [0, 0, 0, 0, 0, 0],
            'Intraventricular': [0, 0, 0, 0, 0, 0],
            'Subarachnoid':     [0, 0, 0, 0, 0, 0],
            'No_Hemorrhage':    [1, 0, 1, 1, 1, 1],
            'file': [f'path/to/49_{i}.nii' for i in range(3)] + [f'path/to/54_{i}.nii' for i in range(3)]
        }
        mock_patients = pd.DataFrame(data)
        mock_load_datasets.return_value = mock_patients

        # 2. Mock Image Loading
        mock_get_images.return_value = (np.random.rand(3, 480, 480), 35.0)

        # 3. Mock Models
        mock_model_instance = MagicMock()
        mock_model_instance.load_patient = MagicMock()

        low_prob = 0.01
        high_prob = 0.99

        def pred(epi, any_ich):
            return {
                'Epidural': epi, 'Subdural': low_prob, 'Intraparenchymal': low_prob,
                'Intraventricular': low_prob, 'Subarachnoid': low_prob, 'Any': any_ich
            }

        # Side effect for slice predictions
        # Called once for inference pass, and again for montage generation
        # We need enough side effects or use a fixed return based on args?
        # Simpler: just cycle or use a large list.
        # Initial Pass: 6 calls.
        # Montage Pass: Random sample of 9 (or fewer).
        # So we need > 6 side effects.

        base_preds = [
            # Patient 49 (Slice 1, 2, 3)
            pred(low_prob, low_prob),   # Slice 1 (GT: Neg)
            pred(high_prob, high_prob), # Slice 2 (GT: Pos Epidural)
            pred(low_prob, low_prob),   # Slice 3 (GT: Neg)
            # Patient 54 (Slice 1, 2, 3)
            pred(low_prob, low_prob),   # Slice 1 (GT: Neg)
            pred(low_prob, low_prob),   # Slice 2 (GT: Neg)
            pred(low_prob, low_prob),   # Slice 3 (GT: Neg)
        ]

        # Extend for montage calls (just repeat)
        mock_model_instance.get_slice_prediction.side_effect = base_preds * 20

        # FIX: Return a list so it can be iterated multiple times (once per patient)
        mock_models.items.return_value = [('CAD_1', mock_model_instance)]

        # Import the main function to run
        from roc_analysis import main

        if os.path.exists('results'):
            shutil.rmtree('results')

        try:
            main()
        except Exception as e:
            self.fail(f"main() raised {e} unexpectedly!")

        # Verify outputs
        self.assertTrue(os.path.exists('results'))

        # Check plots
        self.assertTrue(os.path.exists(os.path.join('results', 'Hssayeni_Slice_Overall_ROC.png')))
        self.assertTrue(os.path.exists(os.path.join('results', 'Hssayeni_Slice_Epidural_ROC.png')))
        self.assertTrue(os.path.exists(os.path.join('results', 'Hssayeni_Patient_Overall_ROC.png')))
        self.assertTrue(os.path.exists(os.path.join('results', 'Hssayeni_Patient_Epidural_ROC.png')))

        # Check Montage Directory
        montage_dir = os.path.join('results', 'Montages')
        self.assertTrue(os.path.exists(montage_dir))
        # Since we mock data perfectly, we might not have FP or FN, but we should have TP/TN for 'Any'
        # P49 Slice 2 is TP. P54 Slices are TN.
        # So we expect Hssayeni_CAD_1_Any_TP_HighSens.png (if thresh < 0.99)
        # Thresh for 90% sens should be roughly 0.99 (since we have clear separation).
        # We should check for at least ONE montage file.
        montage_files = os.listdir(montage_dir)
        self.assertTrue(len(montage_files) > 0, "No montages generated")

        # Check Confusion Matrix Directory
        cm_dir = os.path.join('results', 'Confusion_Matrices')
        self.assertTrue(os.path.exists(cm_dir))
        cm_files = os.listdir(cm_dir)
        self.assertTrue(len(cm_files) > 0, "No confusion matrices generated")

        # Check CSV
        self.assertTrue(os.path.exists(os.path.join('results', 'auc_scores.csv')))
        df = pd.read_csv(os.path.join('results', 'auc_scores.csv'))

        self.assertFalse(df[df['Level'] == 'Slice'].empty, "Slice level results missing from CSV")
        self.assertFalse(df[df['Level'] == 'Patient'].empty, "Patient level results missing from CSV")

        if os.path.exists('results'):
            shutil.rmtree('results')

if __name__ == '__main__':
    unittest.main()

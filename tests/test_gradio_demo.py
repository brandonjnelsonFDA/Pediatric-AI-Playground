import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys
from pathlib import Path

class TestGradioDemo(unittest.TestCase):
    def setUp(self):
        # Create dummy data directories and files
        self.hssayeni_dir = 'dummy_hssayeni'
        self.synth_dir = 'dummy_synth'
        self.model_dir = 'dummy_model'
        os.makedirs(os.path.join(self.hssayeni_dir, 'ct_scans'), exist_ok=True)
        os.makedirs(self.synth_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, 'CAD_1'), exist_ok=True)


        # Dummy Patient_demographics.csv
        with open(os.path.join(self.hssayeni_dir, 'Patient_demographics.csv'), 'w') as f:
            f.write('Patient Number,"Age\n(years)"\n1,10\n')

        # Dummy hemorrhage_diagnosis_raw_ct.csv
        with open(os.path.join(self.hssayeni_dir, 'hemorrhage_diagnosis_raw_ct.csv'), 'w') as f:
            f.write('PatientNumber,SliceNumber,Intraventricular\n1,1,0\n')

        # Dummy synthetic data
        with open(os.path.join(self.synth_dir, 'synth_data.csv'), 'w') as f:
            f.write('subtype,case_id,phantom\nEDH,case_1,10 yr\n')

        # Dummy model file
        Path(os.path.join(self.model_dir, 'CAD_1', 'model.pth')).touch()


    def tearDown(self):
        # Clean up dummy directories and files
        import shutil
        shutil.rmtree(self.hssayeni_dir)
        shutil.rmtree(self.synth_dir)
        shutil.rmtree(self.model_dir)

    def test_load_datasets_adds_dataset_column(self):
        with patch.dict(os.environ, {"HSSAYENI_DIR": "dummy_hssayeni", "SYNTH_DIR": "dummy_synth", "MODEL_PATH": "dummy_model"}):
            with patch('model_utils.InferenceManager') as mock_inference_manager, \
                 patch('matplotlib.pyplot') as mock_pyplot:
                mock_pyplot.subplots.return_value = (MagicMock(), MagicMock())
                # Unload the gradio_demo module if it was imported before
                if 'gradio_demo' in sys.modules:
                    del sys.modules['gradio_demo']
                from gradio_demo import load_datasets
                # Test that the 'dataset' column is added correctly
                patients_df = load_datasets(hssayeni_dir=self.hssayeni_dir, synth_dir=self.synth_dir)
                self.assertIn('dataset', patients_df.columns)
                self.assertEqual(patients_df[patients_df['name'] == 1]['dataset'].iloc[0], 'Hssayeni')
                self.assertEqual(patients_df[patients_df['name'] == 'synthetic 1']['dataset'].iloc[0], 'Synthetic')


    def test_update_patient_dropdown(self):
        with patch.dict(os.environ, {"HSSAYENI_DIR": "dummy_hssayeni", "SYNTH_DIR": "dummy_synth", "MODEL_PATH": "dummy_model"}):
            with patch('model_utils.InferenceManager') as mock_inference_manager, \
                 patch('matplotlib.pyplot') as mock_pyplot:
                mock_pyplot.subplots.return_value = (MagicMock(), MagicMock())
                # Unload the gradio_demo module if it was imported before
                if 'gradio_demo' in sys.modules:
                    del sys.modules['gradio_demo']
                from gradio_demo import update_patient_dropdown

                mock_patients = pd.DataFrame({
                    'age': [10, 20, 30],
                    'dataset': ['Hssayeni', 'Synthetic', 'Hssayeni'],
                    'name': [1, 'synth_1', 2]
                })

                with patch('gradio_demo.patients', mock_patients):
                    # Test filtering by dataset
                    result = update_patient_dropdown(0, 40, ['Hssayeni'])
                    self.assertIn('Patient 1 - Age 10', result['choices'])
                    self.assertNotIn('Patient synth_1 - Age 20', result['choices'])

if __name__ == '__main__':
    unittest.main()

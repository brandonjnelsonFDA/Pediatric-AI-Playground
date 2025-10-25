import os
from pathlib import Path
import requests
import zipfile
import io

import albumentations as A
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np 
import pydicom
import nibabel as nib
import skimage
from monai.transforms import Resize


def resize(phantom: np.ndarray, shape: tuple, **kwargs) -> np.ndarray:
    """Resizes a phantom to a new shape while maintaining aspect ratio.

    This function uses MONAI's Resize transform to resize a 2D or 3D phantom
    array. The `size_mode='longest'` option scales the longest dimension to
    match the corresponding dimension in `shape`, and scales other dimensions
    proportionally.

    mode = 'nearest' is useful for downsizing without interpolation errors

    Args:
        phantom (np.ndarray): The phantom image array to resize.
        shape (tuple): The target shape for the phantom.
        **kwargs: Additional keyword arguments to be passed to
            `monai.transforms.Resize`. 
            E.g.: `from monai.transforms import Resize; Resize?`

    Returns:
        np.ndarray: The resized phantom array.
    """
    resize_transform = Resize(max(shape), size_mode='longest', **kwargs)
    # MONAI transforms expect a channel dimension, so we add and remove one.
    resized = resize_transform(phantom[None])[0]
    return resized

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        x = super(SpatialDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1)
        x = x.squeeze(2)
        return x

class NeuralNet(nn.Module):
    def __init__(self, embed_size=2048*3, LSTM_UNITS=64, DO=0.3):
        super(NeuralNet, self).__init__()
        self.embedding_dropout = SpatialDropout(0.0)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear = nn.Linear(LSTM_UNITS*2, 6)

    def forward(self, x, lengths=None):
        h_embedding = x
        h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd
        output = self.linear(hidden)
        return output

def apply_window(image, center, width):
    """Applies a windowing function to a grayscale image.

    This function is commonly used in medical imaging to adjust the contrast
    and brightness of an image to highlight specific structures.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        center (int): The center of the window.
        width (int): The width of the window.

    Returns:
        np.ndarray: The windowed image.
    """
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


def apply_window_policy(image):
    """Applies a multi-windowing policy to the image.

    This function applies three different windowing functions to the input
    image to highlight brain, subdural, and bone structures, respectively.
    The windowed images are then normalized and stacked to form a 3-channel
    image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The 3-channel image with the windowing policy applied.
    """
    image1 = apply_window(image, 40, 80) # brain
    image2 = apply_window(image, 80, 200) # subdural
    image3 = apply_window(image, 40, 380) # bone
    image1 = (image1 - 0) / 80
    image2 = (image2 - (-20)) / 200
    image3 = (image3 - (-150)) / 380
    image = np.array([
        image1 - image1.mean(),
        image2 - image2.mean(),
        image3 - image3.mean(),
    ]).transpose(1,2,0)
    return image


def classify_images(volume, options, model_path, device):
    """Performs classification on a volume of images using a pre-trained model.

    This function takes a volume of images, extracts features using a
    pre-trained classifier, and then uses a recurrent neural network (RNN) to
    make a final classification.

    Args:
        volume (torch.Tensor): A tensor representing the volume of images, with
            dimensions [num_images, 3, 480, 480].
        options (dict): A dictionary of options. The 'verbose' key can be set
            to True to enable verbose output.
        model_path (str): The path to the pre-trained model file.
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').

    Returns:
        np.ndarray: A NumPy array of classification scores for each image in the
            volume.
    """
    print(options)
    num_images = volume.shape[0]
    if options['verbose']: print('Starting classification')

    # Load and set up classifier
    model_path = Path(model_path)
    classifier_model = torch.load(model_path, weights_only=False, map_location=device) # Load model arch
    classifier_model.fc = nn.Linear(2048, 6) # 6 classes
    classifier_model.to(device)
    # Model was trained with DDP so need this even with 1 GPU
    classifier_model = nn.DataParallel(classifier_model, device_ids=list(range(1)), output_device=device)
    for param in classifier_model.parameters():
        param.requires_grad = False
    classifier_model.load_state_dict(torch.load(next(model_path.parent.glob('model_*.bin')),  map_location=device)) # Load weights

    class Identity(nn.Module):
        """An identity layer that returns its input.

        This layer is used as a placeholder to replace the fully connected
        layer of a pre-trained model, allowing for the extraction of features
        from the penultimate layer.
        """
        def __init__(self):
            super(Identity, self).__init__()
  
        def forward(self, x):
            """Returns the input tensor unchanged.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor, identical to the input.
            """
            return x

    # Extract embedding layers since that's what we need for the LSTM
    classifier_model.module.fc = Identity()
    classifier_model.eval()
    embeddings = np.zeros((num_images, 2048))
    for idx in range(num_images):
        slice = volume[idx, :, :, :].unsqueeze(0).to(device) # maintain first dimension
        out = classifier_model(slice)
        embeddings[idx, :] = out.detach().cpu().numpy().astype(np.float32)

    # Pad out the  embeddings
    lag = np.zeros(embeddings.shape)
    lead = np.zeros(embeddings.shape)
    lag[1:] = embeddings[1:]-embeddings[:-1]
    lead[:-1] = embeddings[:-1]-embeddings[1:]
    embeddings = np.concatenate((embeddings, lag, lead), -1)
    embeddings = torch.from_numpy(embeddings).unsqueeze(0).to(device, dtype=torch.float)

    # INITIALIZE CLASSES FOR LSTM MODEL
    class SpatialDropout(nn.Dropout2d):
        """Applies spatial dropout to a 2D tensor.

        This layer randomly sets entire feature maps to zero, which helps to
        prevent overfitting by encouraging the model to learn more robust
        features.
        """
        def forward(self, x):
            """Applies spatial dropout to the input tensor.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor with spatial dropout applied.
            """
            x = x.unsqueeze(2)    # (N, T, 1, K)
            x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
            x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
            x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
            x = x.squeeze(2)  # (N, T, K)
            return x

    class NeuralNet(nn.Module):
        """A neural network model for classification.

        This model consists of two bidirectional LSTM layers followed by two
        linear layers. It is designed to take a sequence of embeddings as
        input and output a sequence of classification scores.
        """
        def __init__(self, embed_size=2048*3, LSTM_UNITS=64, DO = 0.3):
            super(NeuralNet, self).__init__()

            self.embedding_dropout = SpatialDropout(0.0) #DO)

            self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
            self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

            self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
            self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

            self.linear = nn.Linear(LSTM_UNITS*2, 6)

        def forward(self, x, lengths=None):
            """Performs a forward pass through the network.

            Args:
                x (torch.Tensor): The input tensor, which is a sequence of
                    embeddings.
                lengths (list, optional): A list of sequence lengths. Defaults
                    to None.

            Returns:
                torch.Tensor: The output tensor, which is a sequence of
                    classification scores.
            """
            h_embedding = x

            h_embadd = torch.cat((h_embedding[:,:,:2048], h_embedding[:,:,:2048]), -1)

            h_lstm1, _ = self.lstm1(h_embedding)
            h_lstm2, _ = self.lstm2(h_lstm1)

            h_conc_linear1  = F.relu(self.linear1(h_lstm1))
            h_conc_linear2  = F.relu(self.linear2(h_lstm2))

            hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

            output = self.linear(hidden)

            return output

    # Create model 
    lstm_model = NeuralNet(LSTM_UNITS=2048, DO = 0.3)
    lstm_model = lstm_model.to(device)
    lstm_model.load_state_dict(torch.load(next(model_path.parent.glob('lstm_*.bin')),  map_location=device))
    for param in lstm_model.parameters():
        param.requires_grad = False

    lstm_model.eval()
    if options['verbose']: print('Evaluating embeddings')

    slice_by_slice = False # False is the original implementation for the grand challenge submission
    if slice_by_slice:
        values = np.zeros(shape=(embeddings.shape[1], 6))
        for i in range(embeddings.shape[1]):
            logits = lstm_model(embeddings[:, i, :].unsqueeze(1))
            logits = logits.view(-1, 6)
            temp = torch.sigmoid(logits).detach().cpu().numpy()
            if options['verbose']: print(temp[0][5])
            values[i, :] = temp
    else:   
        logits = lstm_model(embeddings)
        logits = logits.view(-1, 6)
        values = torch.sigmoid(logits).detach().cpu().numpy()

        if options['verbose']:
            for i in range(values.shape[0]):
                print('slice ' + str(i) + ': ' + str(values[i, -1]))
    return values


def predict_image(image, model, device='cuda'):
    """Predicts the classification of a single image.

    This function takes a single image, applies the necessary preprocessing
    steps, and then uses the `classify_images` function to obtain a
    classification.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        model (str): The name of the model to use for classification.
        device (str, optional): The device to run the model on. Defaults to
            'cuda'.

    Returns:
        dict: A dictionary mapping class labels to their predicted scores.
    """
    # image preparation lifted from prepare_images (intended for .dcm and .nii)
    labels = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'Any']
    mean_img = [0.22363983, 0.18190407, 0.2523437 ]
    std_img = [0.32451536, 0.2956294,  0.31335256]
    transform = A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0)

    image = apply_window_policy(image)
    image -= image.min((0,1))
    image = (255*image).astype(np.uint8)
    #cv2.imwrite('test.jpg', image)
    image = resize(image, (480, 480)) # use Monai instead or VITools.resize # skimage.transform.resize changes result!!
    result = transform(image=image)
    image = torch.from_numpy(result["image"])
    image = torch.permute(image, (2, 1, 0)).unsqueeze(0)


    # classify images
    options = {'verbose': True}
    output = classify_images(image, options, model, device=device)
    return dict(zip(labels, output[0]))


def download_and_unzip(url, extract_to='.'):
    """
    Downloads a zip file from a given URL and extracts its contents.

    Args:
        url (str): The URL of the zip file.
        extract_to (str): The directory where the contents should be extracted.
                          Defaults to the current directory.
    """
    try:
        # Download the zip file
        print(f"Downloading from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a BytesIO object from the downloaded content
        zip_file_in_memory = io.BytesIO(response.content)

        # Open the zip file and extract its contents
        with zipfile.ZipFile(zip_file_in_memory, 'r') as zip_ref:
            print(f"Extracting contents to {os.path.abspath(extract_to)}...")
            zip_ref.extractall(extract_to)
        print("Download and extraction complete.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class InferenceManager:
    def __init__(self, model_path, device='cuda'):
        self.model_path = Path(model_path)
        self.device = device
        self.labels = ['EDH', 'IPH', 'IVH', 'SAH', 'SDH', 'Any']
        mean_img = [0.22363983, 0.18190407, 0.2523437]
        std_img = [0.32451536, 0.2956294, 0.31335256]
        self.transform = A.Normalize(mean=mean_img, std=std_img, max_pixel_value=255.0, p=1.0)
        self._load_models()
        self.patient_images = None
        self.patient_predictions = None

    def _load_models(self):
        # Load classifier model
        self.classifier_model = torch.load(self.model_path, weights_only=False, map_location=self.device)
        self.classifier_model.fc = nn.Linear(2048, 6)
        self.classifier_model.to(self.device)
        self.classifier_model = nn.DataParallel(self.classifier_model, device_ids=list(range(1)), output_device=self.device)
        for param in self.classifier_model.parameters():
            param.requires_grad = False
        self.classifier_model.load_state_dict(torch.load(next(self.model_path.parent.glob('model_*.bin')), map_location=self.device))

        self.classifier_model.module.fc = Identity()
        self.classifier_model.eval()

        # Load LSTM model
        self.lstm_model = NeuralNet(LSTM_UNITS=2048, DO=0.3)
        self.lstm_model = self.lstm_model.to(self.device)
        self.lstm_model.load_state_dict(torch.load(next(self.model_path.parent.glob('lstm_*.bin')), map_location=self.device))
        for param in self.lstm_model.parameters():
            param.requires_grad = False
        self.lstm_model.eval()

    def _preprocess_image(self, image):
        image = apply_window_policy(image)
        image -= image.min((0, 1))
        image = (255 * image).astype(np.uint8)
        image = resize(image, (480, 480))
        result = self.transform(image=image)
        image = torch.from_numpy(result["image"])
        image = torch.permute(image, (2, 1, 0)).unsqueeze(0)
        return image

    def _run_inference(self, volume):
        num_images = volume.shape[0]
        embeddings = np.zeros((num_images, 2048))
        for idx in range(num_images):
            slice = volume[idx, :, :, :].unsqueeze(0).to(self.device)
            out = self.classifier_model(slice)
            embeddings[idx, :] = out.detach().cpu().numpy().astype(np.float32)

        lag = np.zeros(embeddings.shape)
        lead = np.zeros(embeddings.shape)
        lag[1:] = embeddings[1:] - embeddings[:-1]
        lead[:-1] = embeddings[:-1] - embeddings[1:]
        embeddings = np.concatenate((embeddings, lag, lead), -1)
        embeddings = torch.from_numpy(embeddings).unsqueeze(0).to(self.device, dtype=torch.float)

        logits = self.lstm_model(embeddings)
        logits = logits.view(-1, 6)
        values = torch.sigmoid(logits).detach().cpu().numpy()
        return values

    def load_patient(self, patient_images):
        self.patient_images = patient_images
        preprocessed_slices = torch.cat([self._preprocess_image(image) for image in self.patient_images])
        self.patient_predictions = self._run_inference(preprocessed_slices)

    def get_slice_prediction(self, slice_num):
        if self.patient_predictions is not None:
            return dict(zip(self.labels, self.patient_predictions[slice_num]))
        return None

    def predict_image_on_the_fly(self, image):
        preprocessed_image = self._preprocess_image(image)
        prediction = self._run_inference(preprocessed_image)
        return dict(zip(self.labels, prediction[0]))
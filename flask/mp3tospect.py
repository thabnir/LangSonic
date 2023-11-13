import numpy as np
import librosa
import resampy
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
from transformers import WhisperFeatureExtractor


class CustomWhisperFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, feature_size=13, **kwargs):
        super().__init__(feature_size=feature_size, **kwargs)


feature_extractor = CustomWhisperFeatureExtractor(
    feature_size=13, sampling_rate=16000, padding_value=0.0, return_attention_mask=False
)


def model_input_from_mp3(
    filepath,
    target_size=(13, 250),
    feature_extractor=feature_extractor,
):
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def img_to_array_in_memory(spec, target_size):
        # Scale the spectrogram as we did before
        scaled_spec = scale_minmax(spec, 0, 255).astype(np.uint8)
        img = np.flip(scaled_spec, axis=0)  # Low frequencies at the bottom
        img_pil = Image.fromarray(img)

        # Resize image if necessary
        if img_pil.size != target_size:
            img_pil = img_pil.resize(target_size[::-1], Image.BICUBIC)

        buffer = BytesIO()
        img_pil.save(buffer, format="PNG")
        buffer.seek(0)

        img_array = image.img_to_array(image.load_img(buffer, color_mode="grayscale"))
        img_array = img_array.reshape((target_size[0], target_size[1], 1))
        img_array = img_array / 255.0  # scale to [0, 1]
        return img_array

    # Load and resample the audio file
    signal, sr = librosa.load(filepath)
    signal = resampy.resample(signal, sr_orig=sr, sr_new=16000, res_type="kaiser_fast")

    # Extract features using the feature extractor
    f = feature_extractor(
        signal,
        sampling_rate=16000,
        padding="max_length",  # pads to 30 seconds
        do_normalize=True,
        feature_size=target_size[0],
        return_attention_mask=False,
    )
    input_features = np.array(f["input_features"])[0]
    input_features = input_features[:, : target_size[1]]

    # Convert the spectrogram to an image array and return
    return img_to_array_in_memory(input_features, target_size)

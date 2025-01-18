
# Assamese CAIR - Low Resource Speech Recognition System

This project implements a low-resource Automatic Speech Recognition (ASR) system using the Wav2Vec 2.0 model for Assamese language. The model is trained on a custom dataset and can transcribe audio files in Assamese. This repository contains the code for training, preprocessing, and evaluating the Wav2Vec 2.0 model, fine-tuned for Assamese language.

## Project Structure

- `training.py`: Script to train the model with a custom dataset.
- `structure.toml`: Configuration file to define hyperparameters, paths, and dataset settings.
- `metric.py`: Metric calculations for evaluating model performance, including Word Error Rate (WER).
- `progress_bar.py`: Custom progress bar implementation for training and validation loops.
- `tensorboard.py`: TensorBoard writer for logging training progress.
- `head_model.py`: Defines the custom head model added to Wav2Vec 2.0.
- `train.py`: Main training loop and validation logic.
- `data_collector.py`: Data collator that handles padding and processing of input batches.
- `torch_dataset.py`: Custom dataset class to load audio files and transcriptions.
- `module_initialization.py`: Module initialization utility for dynamically loading classes.
- `features.py`: Helper functions for feature extraction from audio files.
- `utils/`: Utility scripts for various tasks like logging and progress tracking.

## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Torchaudio
- tqdm
- pandas
- tensorboard
- Evaluate

You can install the necessary dependencies using pip:

```
pip install torch torchaudio transformers tqdm pandas tensorboard evaluate
```

## Configuration

The `structure.toml` file contains all the configuration settings for the training process. It defines:

- `project_name`: Name of the project.
- `epochs`: Number of epochs for training.
- `grad_accm_steps`: Gradient accumulation steps for large batches.
- `save_dir`: Directory where model checkpoints will be saved.
- `pretrained_model`: Path to the pre-trained Wav2Vec 2.0 model.
- `sr`: Sampling rate of the audio data.
- `training_data`: Path to the training dataset and data-related settings.
- `validation_data`: Path to the validation dataset and validation settings.
- `test_data`: Path to the test dataset.
- `optimizer`: Optimizer settings including learning rate.
- `training`: Settings related to training, like validation intervals.

## Training

To train the model, run the following command:

```
python training.py --structure path_to_structure.toml
```

This will start the training process using the configurations provided in the `structure.toml` file.

### Training Configuration

The training process includes:

- Loading the dataset using the `torch_dataset.py` class.
- Initializing the model using `Wav2Vec2ForCTC` and a custom head defined in `head_model.py`.
- Using the `Trainer` class in `train.py` to handle training and validation.
- Logging metrics such as loss and WER (Word Error Rate) to TensorBoard and updating progress with `progress_bar.py`.

### Checkpoints

The model will save checkpoints during training to the `save_dir` specified in `structure.toml`. The best model is saved as `best_model.tar`.

## Evaluation

After training, you can evaluate the model using the test set. The model uses the WER metric for evaluation, which is calculated using the `metric.py` class.

## Customization

You can modify the following to fit your specific use case:

- **Dataset**: Provide your own dataset in the `structure.toml` file under the `training_data`, `validation_data`, and `test_data` sections.
- **Model**: You can use any pre-trained model available in Hugging Face’s model hub by specifying it in the `pretrained_model` section.


## Acknowledgments

- The model is based on the Wav2Vec 2.0 architecture by Facebook AI.
- Special thanks to Hugging Face for providing the transformers library.
```

2. **Create the file**:
   - Open any text editor (e.g., Notepad, VSCode, Sublime Text).
   - Paste the content above into the editor.
   - Save the file as `README.md`.

3. **Optionally, you can upload it to a platform like GitHub**:
   If you want to share it or use it in a repository, upload this `README.md` file to your repository on GitHub.

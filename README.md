
# ASSAMESE_CAIR: Low Resource Speech Recognition

This project implements a low-resource speech recognition system using Wav2Vec2, targeting the Assamese language. It utilizes pre-trained models, fine-tuned on a custom dataset, and evaluates the model's performance using Word Error Rate (WER). The system is designed to be flexible and can easily be extended to other languages and tasks.

## Project Structure

The project consists of multiple modules that manage various parts of the pipeline, including dataset loading, training, evaluation, and logging. The main components are:

- **Model and Tokenizer**: Wav2Vec2 for feature extraction and CTC for speech-to-text.
- **Dataset Handling**: Custom dataset class and data collators to handle speech data efficiently.
- **Training**: Custom training loop with features like gradient accumulation, validation, and checkpoint saving.
- **Metrics and Logging**: WER evaluation, TensorBoard logging, and a custom progress bar.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Huggingface Transformers
- torchaudio
- tqdm
- evaluate
- pandas

To install the required dependencies, use:

```
pip install -r requirements.txt
```

## Directory Structure

- `training.py`: Main script for training the Wav2Vec2-based model.
- `structure.toml`: Configuration file containing paths, hyperparameters, and training settings.
- `metric.py`: Metric calculation, including Word Error Rate (WER).
- `progress_bar.py`: Custom progress bar for tracking training progress.
- `tensorboard.py`: TensorBoard logging functionality.
- `head_model.py`: Defines a custom head for the Wav2Vec2 model.
- `train.py`: Contains the training loop and validation logic.
- `data_collector.py`: Data collator for padding and preparing batches.
- `torch_dataset.py`: Dataset class for loading and processing audio files.
- `module_initialization.py`: Handles dynamic initialization of modules based on configuration.
- `features.py`: Utility functions for audio feature extraction.

## How to Run

### Training the Model

To start training, run the following command:

```
python training.py -s path_to_structure_file
```

This will initiate training with the configuration specified in the provided TOML file.

- `-s`: Path to the `structure.toml` file containing the configuration.

### Configuration (`structure.toml`)

The `structure.toml` file contains all the necessary paths and parameters required for training. The main sections include:

- **Main**: Defines the project name, number of epochs, gradient accumulation steps, save directories, and the sample rate.
- **Training Data**: Specifies the path to the training dataset and the special tokens for the tokenizer.
- **Validation Data**: Specifies the path to the validation dataset.
- **Test Data**: Defines the test data for evaluation.
- **Pretrained Model**: Specifies the path to the pretrained Wav2Vec2 model.
- **Optimizer**: Defines the learning rate for the optimizer.
- **Training**: Specifies parameters like validation intervals and saving conditions.

### Example Structure:

```toml
[main]
project_name = "ASSAMESE_CAIR"
epochs = 100
grad_accm_steps = 2
save_dir = "saved/"
pretrained_model = "facebook/wav2vec2-base"
sr = 16000

[training_data]
path = "data_process.dataset.Dataset_creation"
    [training_data.args]
    path = "/wav2vec2_assamese/datasets_NEW/train.csv"
    sr = 16000
    
        [training_data.args.special_tokens]
        bos_token = "<bos>"
        eos_token = "<eos>"
        unk_token = "<unk>"
        pad_token = "<pad>"
    [training_data.dataloader]
    batch_size = 4
    pin_memory = true
    drop_last = true

[validation_data]
path = "data_process.dataset.Dataset_creation"
    [validation_data.args]
    path = "/wav2vec2_assamese/datasets_NEW/validation.csv"
    sr = 16000

        [validation_data.args.special_tokens]
        bos_token = "<bos>"
        eos_token = "<eos>"
        unk_token = "<unk>"
        pad_token = "<pad>"
    [validation_data.dataloader]
    batch_size = 16

[test_data]
path = "data_process.dataset.Dataset_creation"
    [test_data.args]
    path = "/wav2vec2_assamese/datasets_NEW/test.csv"
    sr = 16000

[pretrained_model]
path = "/wav2vec2_assamese/pretrained/"

[optimizer]
lr = 1e-4

[training]
path = "training_module.train.Train"
    [training.args]
    validation_interval = 100
    save_max_metric_score = false
```

## Model Training Process

### Dataset

The model expects a CSV file containing columns for the `audio_path` and `transcription` (text). The audio files should be in WAV format. The dataset is loaded and processed using the `Dataset` class, which extracts features from the audio files, applies the necessary transformations, and prepares them for training.

### Training Loop

The model is trained using the `Train` class, which handles the entire training loop:

1. **Forward Pass**: The model processes the input features and generates predictions.
2. **Loss Calculation**: The CTC loss is calculated and backpropagated.
3. **Optimizer Step**: The model weights are updated based on the gradients.
4. **Validation**: The model is evaluated on the validation dataset at specified intervals.
5. **Checkpointing**: Model checkpoints are saved after each epoch, and the best model is saved based on the validation WER.

### Metrics

The primary evaluation metric is **Word Error Rate (WER)**, which measures the accuracy of the transcriptions predicted by the model.

## Logging

Training progress and metrics are logged using **TensorBoard**. A custom progress bar is displayed during training, showing the current training loss, learning rate, and WER.

To visualize the logs, run:

```bash
tensorboard --logdir=path_to_log_dir
```

## Conclusion

This project provides a flexible and extendable framework for training speech recognition models using Wav2Vec2. By modifying the configuration file, you can fine-tune the model on your own datasets and evaluate its performance.

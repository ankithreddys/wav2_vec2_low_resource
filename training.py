import pandas as pd
import toml
import argparse
import os
import torch
import datetime
import warnings
warnings.filterwarnings("ignore")

from time import gmtime, strftime
from torch.utils.data import DataLoader
from general_init.module_initialization import initialize_module
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from torch_dataloading.data_collator import DataCollatorCTCWithPadding
from training_module.head_model import CustomHead
from utils.metric import Metric
import os


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)
    torch.cuda.empty_cache()
    #print(torch.cuda.max_memory_allocated())
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:20480"

    save_dir = os.path.join(structure["main"]["save_dir"],structure["main"]["project_name"]+'/checkpoints')
    log_dir = os.path.join(structure["main"]["save_dir"],structure["main"]["project_name"]+'/logs')  

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    structure_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
    with open(os.path.join(structure["main"]["save_dir"], structure["main"]['project_name'] + '/' + structure_name), 'w+') as f:
        toml.dump(structure, f)
        f.close()
    

    tokenizer = Wav2Vec2CTCTokenizer(
                                    vocab_file="/wav2vec2_assamese/DATASETS/vocab_assamese_new.json", 
                                    bos_token = structure["training_data"]["args"]["special_tokens"]['bos_token'],
                                    eos_token = structure["training_data"]["args"]["special_tokens"]['eos_token'],
                                    unk_token = structure["training_data"]["args"]["special_tokens"]['unk_token'],
                                    pad_token = structure["training_data"]["args"]["special_tokens"]['pad_token'],
                                    word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size = 1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    default_collate = DataCollatorCTCWithPadding(processor, structure['main']['sr'], padding = True)
    structure["training_data"]['args']['processor'] = processor
    train_module_init = initialize_module(structure['training_data']['path'],structure['training_data']['args'])
    train_ds = train_module_init.dataset_loading()
    train_dl = DataLoader(train_ds, **structure["training_data"]["dataloader"],collate_fn = default_collate)
    #print(torch.cuda.max_memory_allocated())
    #a=input("enter anything?")
    train_module_init.vocab_dict()

    structure['validation_data']['args']['processor'] = processor
    val_module_init = initialize_module(structure["validation_data"]["path"],args = structure["validation_data"]["args"])
    valid_ds = val_module_init.dataset_loading()
    valid_dl = DataLoader(valid_ds, **structure["validation_data"]["dataloader"],collate_fn = default_collate)

    print("Number of training instances: ", len(train_ds))
    print("Number of validation instances: ", len(valid_ds))

    metric_wer = Metric(processor)
    
    model = Wav2Vec2ForCTC.from_pretrained(
        structure['main']['pretrained_model'], 
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        gradient_checkpointing=False
    )
    #print(model)
    #print('####################################################################')
    for name, param in model.named_parameters():
            param.requires_grad = False
    custom_head = CustomHead(model.config.hidden_size, 512, 1005) 
    model.lm_head = custom_head
    
    #print(model)
    # model.lm_head.to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        params = model.parameters()
    )

    steps_per_epoch = (len(train_dl)//structure["main"]["grad_accm_steps"]) + (len(train_dl)%structure["main"]["grad_accm_steps"] != 0)

    trainer_class = initialize_module(structure["training"]["path"],structure["training"]["args"], initialize=False)
    trainer = trainer_class(
        #dist = dist,
        #rank = rank,
        #n_gpus = len(structure["main"]["cuda_device_ids"]),
        device = device,
        structure = structure,
        epochs = structure["main"]["epochs"],
        steps_per_epoch = steps_per_epoch,
        model = model,
        metric_wer = metric_wer,
        processor = processor,
        train_dl = train_dl,
        valid_dl = valid_dl,
        #train_sampler = train_sampler,
        #val_sampler = val_sampler,
        optimizer = optimizer,
        save_dir = save_dir,
        log_dir = log_dir,
        grad_accm_steps = structure["main"]["grad_accm_steps"]
        #use_amp = use_amp,
        #max_clip_grad_norm = max_clip_grad_norm
    )
    trainer.train()




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Pass these arguments when you start training for Wav2vec2 Model')
    args.add_argument('-s', '--structure', required=True, type=str,
                      help='structure file path (default: None)')
   
    args = args.parse_args()
    structure = toml.load(args.structure)
    main()
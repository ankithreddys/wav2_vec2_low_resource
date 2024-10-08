from evaluate import load
import torch

class Metric:
    def __init__(self, processor):
        self.processor = processor
        self.wer_metric = load("wer")
        self.i = 0
    def __call__(self, logits, labels):
        self.i += 1
        preds = torch.argmax(logits, axis=-1)

        labels[labels == -100] = self.processor.tokenizer.pad_token_id

        pred_strs = self.processor.batch_decode(preds)
        # we do not want to group tokens when computing the metrics
        label_strs = self.processor.batch_decode(labels, group_tokens=False)
        if self.i == 25:
            print(f'pred_strs:{pred_strs}')
            print('##################################################################')
            print(f'label_strs:{label_strs}')
            self.i = 0

        wer = self.wer_metric.compute(predictions=pred_strs, references=label_strs)

        return wer
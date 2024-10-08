from typing import Dict
from tqdm import tqdm

class Pro_BAR:
    def __init__(self, n_steps: int, width: int, stateful_metrics: list = None):
        self.n_steps = n_steps
        self.width = width
        self.stateful_metrics = stateful_metrics or []
        self.pbar = tqdm(total=n_steps, ncols=width)

    def update(self, step: int, scores: Dict[str, float]) -> None:
        if not isinstance(scores, dict):
            raise TypeError("scores must be a dictionary")

        formatted_scores = [(f"{k}", v) for k, v in scores.items()]
        print(formatted_scores)
        self.pbar.set_postfix(formatted_scores)
        self.pbar.update(1)
        '''for metric_name in self.stateful_metrics:
            if scores[metric_name]:
                metric_value = scores[metric_name]
                self.pbar.set_postfix({metric_name: metric_value})
            else:
                pass
'''
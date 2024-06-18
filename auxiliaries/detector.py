"""
This class outline the detector behaviours.
"""
from collections import Counter

import torch
import torch.nn as nn
from torch.autograd import Variable


class Detector:
    def __init__(self, model, options):
        self.device = options["device"]
        self.model = model
        self.model_path = options["model_path"]
        self.num_candidates = options["num_candidates"]
        self.num_classes = options["num_classes"]
        self.input_size = options["input_size"]

    """
    load the pytorch model generated in the training phase
    """
    def load_model(self):
        model = self.model.to(self.device)
        try:
            model.load_state_dict(torch.load(self.model_path)["state_dict"])
        except FileNotFoundError as err:
            print("Model not found, cannot load model")
            return None
        model.eval()
        print("model_path: {}".format(self.model_path))
        return model

    """
    Detect the anomaly using the loaded model, based on predicting the label values, with a number of candidates specified in the config.
    @parm model: pytorch model
    @parm seq: the sequence of event 
    @parm label: the current event 
    """
    def detect_anomaly(self, model, seq, label):
        with torch.no_grad():
            seq1 = [0] * self.num_classes  #
            log_conuter = Counter(seq)  #
            for key in log_conuter:
                seq1[key] = log_conuter[key]

            seq = (
                torch.tensor(seq, dtype=torch.float)
                .view(-1, len(seq), self.input_size)
                .to(self.device)
            )
            seq1 = (
                torch.tensor(seq1, dtype=torch.float)
                .view(-1, self.num_classes, self.input_size)
                .to(self.device)
            )
            label = torch.tensor(label).view(-1).to(self.device)
            output = model(features=[seq, seq1], device=self.device)
            predicted = torch.argsort(output, 1)[0][-self.num_candidates :]
            if label not in predicted:
                return True
            else:
                return False

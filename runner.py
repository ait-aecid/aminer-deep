import sys
import json
import time
import os
import zmq

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#from auxiliaries.detector import Detector
#from auxiliaries.utils import *
#from models.lstm import deeplog
#from auxiliaries.trainer import Trainer
#from config import *
#from auxiliaries.sample import *

#def train(model, data, labels):
#    print('Start training...')
#    trainer = Trainer(model, options)
#    trainer.start_train(data, labels)
#    print('Finished training!')

testdata = ["aminer:('blk_-1608999687919862906',):0:0",
            "aminer:('blk_-1608999687919862906',):0:1",
            "aminer:('blk_-1608999687919862906',):0:0",
            "aminer:('blk_-1608999687919862906',):0:0",
            "aminer:('blk_-1608999687919862906',):0:2",
            "aminer:('blk_-1608999687919862906',):0:2",
            "aminer:('blk_-1608999687919862906',):0:3",
            "aminer:('blk_-1608999687919862906',):0:3",
            "aminer:('blk_-1608999687919862906',):0:2",
            "aminer:('blk_-1608999687919862906',):0:3",
            "aminer:('blk_-1608999687919862906',):0:4"]

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=100):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        out = self.softmax(out)
        return out
    
    def update_output_dim(self, new_output_dim):
        self.output_dim = new_output_dim
        self.fc = nn.Linear(self.hidden_dim, new_output_dim)
        self.softmax = nn.Softmax(dim=1)

def incremental_train(model, data, labels, criterion, optimizer, batch_size=32, epochs=1):
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i+batch_size]
            y_batch = labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

def detect_anomalies(model, data, true_last_elements, top_k=20):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            input_seq = data[i].unsqueeze(0)
            predictions = model(input_seq)
            top_k_predictions = torch.topk(predictions, top_k).indices.squeeze(0).tolist()
            #print(top_k_predictions)
            
            if true_last_elements[i].item() not in top_k_predictions:
                anomalies.append(True)
            else:
                anomalies.append(False)
    
    return np.array(anomalies)

# Helper function to update model for new categorical values
def update_model_for_new_categories(model, data):
    current_output_dim = model.output_dim
    max_value = data.max().item()
    if max_value >= current_output_dim:
        new_output_dim = max_value + 1
        model.update_output_dim(new_output_dim)

if __name__ == "__main__":
    #model = deeplog(
    #        input_size=options["input_size"],
    #        hidden_size=options["hidden_size"],
    #        num_layers=options["num_layers"],
    #        num_keys=options["num_classes"],
    #    )
    #trainer = Trainer(dl_model, options)
    #model = trainer.load_model()
    window = 10
    num_predicted = 10
    save_path = 'lstm_predictor.pth'
    seqs = {}
    context = zmq.Context()
    zmq_pub_endpoint = "tcp://127.0.0.1:5559"
    zmq_sub_endpoint = "tcp://127.0.0.1:5560"
    zmq_aminer_top = "aminer"
    zmq_detector_top = "deep-aminer"
    zmq_pub_socket = context.socket(zmq.PUB)
    zmq_pub_socket.connect(zmq_pub_endpoint)
    zmq_sub_socket = context.socket(zmq.SUB)
    zmq_sub_socket.connect(zmq_sub_endpoint)
    zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, zmq_aminer_top)
    #data = {'Sequentials': []}
    #labels = []
    train_data = []
    unique_events = set() # for output_dim
    last_num_unique_events = 0
    model = None
    try:
        i = 0
        while True:
            print("Waiting for the AMiner output.......")
            msg = testdata[i] #zmq_sub_socket.recv_string()
            #print(msg)
            i += 1
            #if msg == options["zmq_aminer_top"] + ':connection_ack1':
            #    zmq_pub_socket.send_string("{}:{}".format(options["zmq_detector_top"], "connection_ack2"))
            top, group, learn_mode, event = msg.split(":")
            #seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
            #seq_list = [[int(x)] for x in seq] # Trainer expects data in this format
            #seq = [int(x) for x in seq]
            unique_events.add(event)
            if group not in seqs:
                seqs[group] = [[int(event)]]
            else:
                seqs[group].append([int(event)])
            if len(seqs[group]) > window:
                #result_line = [group, seq[:-1], seq[-1], False]
                seqs[group] = seqs[group][-window:]
                if learn_mode == "True":
                    train_data.append(seqs[group]) #[:-1])
                    #train_labels.append(seqs[group][-1])
                    #data.append((seq[:-1], seq[-1]))
                    print(seqs[group])
                else:
                    if model is None:
                        model = LSTMPredictor(output_dim=len(unique_events))
                        if os.path.exists(save_path):
                            print('Load model from ' + save_path)
                            model.load_state_dict(torch.load(save_path))
                        else:
                            print('No model found, create new model')
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=0.001)
                    elif last_num_unique_events != len(unique_events):
                        model.update_output_dim(len(unique_events))
                    last_num_unique_events = len(unique_events)
                    if train_data:
                        data_tensor = torch.FloatTensor(train_data)
                        input_sequences = data_tensor[:, :-1, :]
                        labels = data_tensor[:, -1, :].squeeze(-1).long()
                        incremental_train(model, input_sequences, labels, criterion, optimizer)
                        data = []
                    data_tensor = torch.FloatTensor([seqs[group]])
                    input_sequences = data_tensor[:, :-1, :]
                    labels = data_tensor[:, -1, :].squeeze(-1).long()
                    anomalies = detect_anomalies(model, input_sequences, labels, top_k=min(num_predicted, len(unique_events)))
                    #result = detector.detect_anomaly(model, seq[:-1], seq[-1])
                    #result_line[3] = result
                    #print("Sending the detector result: {}".format(result_line))
                    #zmq_pub_socket.send_string("{}:{}".format(zmq_detector_top, json.dumps(result_line)))
                    print(anomalies)
    except KeyboardInterrupt:
        zmq_pub_socket.close()
        zmq_sub_socket.close()
        context.term()
        if data:
            print('KeyboardInterrupt received - saving model...')
            torch.save(model.state_dict(), save_path)

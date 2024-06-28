import sys
import json
import time
import os
import zmq
import copy

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

def incremental_train(model, data, labels, criterion, optimizer, batch_size=32): #, epochs=1):
    model.train()
    #for epoch in range(epochs):
    losses = []
    #print('start batch')
    for i in range(0, len(data), batch_size):
        x_batch = data[i:i+batch_size]
        #print(x_batch)
        y_batch = labels[i:i+batch_size]
        #print(y_batch)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses

def detect_anomalies(model, data, true_last_elements, top_k=20):
    model.eval()
    anomalies = []
    with torch.no_grad():
        if True:
        #for i in range(data.shape[0]):
            input_seq = data[0].unsqueeze(0)
            predictions = model(input_seq)
            top_k_predictions = torch.topk(predictions, top_k).indices.squeeze(0).tolist()
            #print(str(input_seq) + ' --> ' + str(top_k_predictions.index(true_last_elements[i].item())))
            
            if true_last_elements[0].item() not in top_k_predictions:
                #print(str(input_seq.tolist()) + ' -->' + str(top_k_predictions) + ' --> -1')
                ind = torch.topk(predictions, len(predictions.tolist()[0])).indices.squeeze(0).tolist().index(true_last_elements[0].item())
                anomalies.append(True)
            else:
                #print(str(input_seq.tolist()) + ' --> ' + str(top_k_predictions) + ' --> ' + str(top_k_predictions.index(true_last_elements[0].item())))
                ind = top_k_predictions.index(true_last_elements[0].item())
                anomalies.append(False)
            debug = {}
            for k in top_k_predictions:
                debug[k] = round(float(predictions.tolist()[0][k]), 2)
            #print(str(input_seq.tolist()) + ' --> ' + str(debug.items()) + ' --> ' + str(ind))
   
    return np.array(anomalies), str(input_seq.tolist()) + ' --> ' + str(debug.items()) + ' --> correct would be ' + str(true_last_elements[0].item()) + ' at pos ' + str(ind)

# Helper function to update model for new categorical values
def update_model_for_new_categories(model, data):
    current_output_dim = model.output_dim
    max_value = data.max().item()
    if max_value >= current_output_dim:
        new_output_dim = max_value + 1
        model.update_output_dim(new_output_dim)

if __name__ == "__main__":
    anom = set()
    with open('anomaly_label.csv') as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.split(',')
            if parts[1].startswith("Anomaly"):
                anom.add('(\'' + parts[0] + '\',)')
    print(len(anom))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    detected = set()
    all_groups = set()
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
    batch_size = 32
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
    unique_events = set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) # set() # for output_dim
    last_num_unique_events = len(unique_events)
    model = None
    print('Lets go')
    try:
        i = 0
        while True:
            #print("Waiting for the AMiner output.......")
            #msg = testdata[i]
            msg = zmq_sub_socket.recv_string()
            #print(msg)
            i += 1
            #if msg == options["zmq_aminer_top"] + ':connection_ack1':
            #    zmq_pub_socket.send_string("{}:{}".format(options["zmq_detector_top"], "connection_ack2"))
            top, group, learn_mode, event = msg.split(":")
            #seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
            #seq_list = [[int(x)] for x in seq] # Trainer expects data in this format
            #seq = [int(x) for x in seq]
            unique_events.add(int(event))
            if group not in seqs:
                seqs[group] = [[-1]] * window + [[int(event)]]
            else:
                seqs[group].append([int(event)])
            #if len(seqs[group]) > window:
            #result_line = [group, seq[:-1], seq[-1], False]
            seqs[group] = seqs[group][-window:]
            #print(len(seqs[group]))
            #for group, seq in seqs.items():
            #    print(str(group) + ': ' + str(len(seq)))
            #print("")
            if learn_mode == "1":
                train_data.append(copy.deepcopy(seqs[group])) #[:-1])
                #train_labels.append(seqs[group][-1])
                #data.append((seq[:-1], seq[-1]))
                #print(seqs[group])
            #else:
            if True:
                if model is None:
                    #model = LSTMPredictor(output_dim=len(unique_events))
                    if os.path.exists(save_path):
                        print('Load model from ' + save_path)
                        state_dict = torch.load(save_path)
                        fc_weight = state_dict['fc.weight']
                        output_dim = fc_weight.shape[0]
                        model = LSTMPredictor(output_dim=output_dim)
                        model.load_state_dict(state_dict)
                    else:
                        model = LSTMPredictor(output_dim=len(unique_events))
                        print('No model found, create new model')
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                elif last_num_unique_events != len(unique_events):
                    print('UPDATE!!! ' + str(last_num_unique_events) + ' ' + str(len(unique_events)))
                    model.update_output_dim(len(unique_events))
                last_num_unique_events = len(unique_events)
                if train_data and len(train_data) > batch_size:
                    #print("train " + str(train_data))
                    data_tensor = torch.FloatTensor(train_data)
                    input_sequences = data_tensor[:, :-1, :]
                    labels = data_tensor[:, -1, :].squeeze(-1).long()
                    losses = incremental_train(model, input_sequences, labels, criterion, optimizer, batch_size)
                    #print(losses)
                    train_data = []
                #print([seqs[group]])
                data_tensor = torch.FloatTensor([seqs[group]])
                input_sequences = data_tensor[:, :-1, :]
                labels = data_tensor[:, -1, :].squeeze(-1).long()
                anomalies, debugstr = detect_anomalies(model, input_sequences, labels, top_k=min(num_predicted, len(unique_events)))
                all_groups.add(group)
                #print(anomalies)
                if anomalies[0] == True:
                    detected.add(group)
                    if group not in anom:
                        #print(debugstr)
                        pass
                    #result = detector.detect_anomaly(model, seq[:-1], seq[-1])
                    #result_line[3] = result
                    #print("Sending the detector result: {}".format(result_line))
                    #zmq_pub_socket.send_string("{}:{}".format(zmq_detector_top, json.dumps(result_line)))
                    #print(anomalies)
                if i % 5000 == 0:
                    i += 1
                    print(i)
                    for group in detected:
                        if group in anom:
                            tp += 1
                        else:
                            fp += 1
                    for group in all_groups.difference(detected):
                        if group in anom:
                            fn += 1
                        else:
                            tn += 1
                    if tp + fp + fn > 0:
                        print('tp=' + str(tp) + ', tn=' + str(tn) + ', fp=' + str(fp) + ', fn=' + str(fn) + ', tpr=' + str(tp / (tp + fn)) + ', fpr=' + str(fp / (fp + tn)) + ', f1=' + str(tp / (tp + 0.5 * (fp + fn))))
                    #else:
                    #    print('NaN')
                    tp = 0
                    fp = 0
                    tn = 0
                    fn = 0
                    detected = set()
                    all_groups = set()
                zmq_pub_socket.send_string("{}:{}".format(zmq_detector_top, "1"))
    except KeyboardInterrupt:
        zmq_pub_socket.close()
        zmq_sub_socket.close()
        context.term()
        #if data:
        #print('KeyboardInterrupt received - saving model...')
        torch.save(model.state_dict(), save_path)

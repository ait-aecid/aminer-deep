import sys
import json
import time

import zmq
from auxiliaries.detector import Detector
from auxiliaries.utils import *
from models.lstm import deeplog
from auxiliaries.trainer import Trainer
from config import *
from auxiliaries.sample import *

def train(model, data):
    print('Start training...')
    trainer = Trainer(model, options)
    trainer.start_train(data)
    print('Finished training!')

def start_detector(detector, model, zmq_pub_socket, zmq_sub_socket):
    data = []
    while True:
        print("Waiting for the AMiner output.......")
        msg = zmq_sub_socket.recv_string()
        print(msg)
        top, group, learn_mode, seq = msg.split(":")
        seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
        seq = [int(x) for x in seq]
        result_line = [group, seq[:-1], seq[-1], False]
        
        if learn_mode:
            print(seq)
            data.append(seq)
        else:
            if data:
                train(model, data)
                data = []
            result = detector.detect_anomaly(model, seq[:-1], seq[-1])
            result_line[3] = result
            print("Sending the detector result: {}".format(result_line))
            zmq_pub_socket.send_string("{}:{}".format(options["zmq_detector_top"], json.dumps(result_line)))

if __name__ == "__main__":
    Model = deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            num_keys=options["num_classes"],
        )
    detector = Detector(Model, options)
    model = detector.load_model()
    context = zmq.Context()
    zmq_pub_socket = context.socket(zmq.PUB)
    zmq_pub_socket.connect(options["zmq_pub_endpoint"])
    zmq_sub_socket = context.socket(zmq.SUB)
    zmq_sub_socket.connect(options["zmq_sub_endpoint"])
    zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, options["zmq_aminer_top"])
    try:
        start_detector(detector, model, zmq_pub_socket, zmq_sub_socket)
    except KeyboardInterrupt:
        zmq_pub_socket.close()
        zmq_sub_socket.close()
        context.term()
        if data:
            print('KeyboardInterrupt received - train model with latest data...')
            train()

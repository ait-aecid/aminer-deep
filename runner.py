import sys
import json
import time

import zmq
#from auxiliaries.detector import Detector
from auxiliaries.detector import Detector
from auxiliaries.utils import *
from models.lstm import deeplog
from auxiliaries.trainer import Trainer
from config import *
from auxiliaries.sample import *

def train(model, data, labels):
    print('Start training...')
    trainer = Trainer(model, options)
    trainer.start_train(data, labels)
    print('Finished training!')

testdata = ["aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, -1, -1, -1, -1, -1, 0]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, -1, -1, -1, -1, 0, 1]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, -1, -1, -1, 0, 1, 0]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, -1, -1, 0, 1, 0, 0]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, -1, 0, 1, 0, 0, 2]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, -1, 0, 1, 0, 0, 2, 2]",
            "aminer:('blk_-1608999687919862906',):True:[-1, -1, -1, 0, 1, 0, 0, 2, 2, 3]",
            "aminer:('blk_-1608999687919862906',):False:[-1, -1, 0, 1, 0, 0, 2, 2, 3, 3]"]

if __name__ == "__main__":
    model = deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            num_keys=options["num_classes"],
        )
    #trainer = Trainer(dl_model, options)
    #model = trainer.load_model()
    context = zmq.Context()
    zmq_pub_socket = context.socket(zmq.PUB)
    zmq_pub_socket.connect(options["zmq_pub_endpoint"])
    zmq_sub_socket = context.socket(zmq.SUB)
    zmq_sub_socket.connect(options["zmq_sub_endpoint"])
    zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, options["zmq_aminer_top"])
    data = {'Sequentials': []}
    labels = []
    try:
        i = 0
        while True:
            print("Waiting for the AMiner output.......")
            msg = zmq_sub_socket.recv_string()
            i += 1
            top, group, learn_mode, seq = msg.split(":")
            seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
            seq_list = [[int(x)] for x in seq] # Trainer expects data in this format
            seq = [int(x) for x in seq]
            result_line = [group, seq[:-1], seq[-1], False]
    
            if learn_mode == "True":
                data['Sequentials'].append(seq_list[:-1])
                labels.append(seq[-1])
                #data.append((seq[:-1], seq[-1]))
            else:
                if data:
                    train(model, data, labels)
                    data = []
                    labels = []
                    detector = Detector(model, options)
                result = detector.detect_anomaly(model, seq[:-1], seq[-1])
                result_line[3] = result
                print("Sending the detector result: {}".format(result_line))
                zmq_pub_socket.send_string("{}:{}".format(options["zmq_detector_top"], json.dumps(result_line)))
    except KeyboardInterrupt:
        zmq_pub_socket.close()
        zmq_sub_socket.close()
        context.term()
        if data:
            print('KeyboardInterrupt received - train model with latest data...')
            train(model, data, labels)

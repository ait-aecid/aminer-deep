import argparse
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

import sys
sys.path.append('../')

seed_everything(seed=1234)

def do_analyze():
    print("Loading sequences ....")
    logs_normal, groups_normal = load_sequences(options['data_save_dir'] + options['data_file_name'] + "_normal")
    logs_abnormal, groups_abnormal = load_sequences(options['data_save_dir'] + options['data_file_name'] + "_abnormal")
    logs_normal_train, groups_normal_train = load_sequences(options['data_save_dir'] + options['data_file_name'] + "_normal_train")
    logs_normal_test, groups_normal_test = load_sequences(options['data_save_dir'] + options['data_file_name'] + "_normal_test")
    normal_unique = np.unique(logs_normal['Sequentials'], axis=0)
    abnormal_unique = np.unique(logs_abnormal['Sequentials'], axis=0)
    normal_train_unique = np.unique(logs_normal_train['Sequentials'], axis=0)
    normal_test_unique = np.unique(logs_normal_test['Sequentials'], axis=0)
    print("Total number of normal sequences: {}".format(len(logs_normal['Sequentials'])))
    print("Total number of abnormal sequences               : {}".format(len(logs_abnormal['Sequentials'])))
    print("Total number of normal training sequences        : {}".format(len(logs_normal_train['Sequentials'])))
    print("Total number of normal testing sequences         : {}".format(len(logs_normal_test['Sequentials'])))
    print("Total number of unique normal sequences           : {}".format(len(normal_unique)))
    print("Total number of unique abnormal sequences         : {}".format(len(abnormal_unique)))
    print("Total number of unique normal training sequences  : {}".format(len(normal_train_unique)))
    print("Total number of unique normal testing sequences   : {}".format(len(normal_test_unique)))
    print("Event types total in normal                      : {}, including : {}".format(len(np.unique(normal_unique)), np.unique(normal_unique)))
    print("Event types total in abnormal                    : {}, including : {}".format(len(np.unique(abnormal_unique)), np.unique(abnormal_unique)))
    print("Event types total in normal training             : {}, including : {}".format(len(np.unique(normal_train_unique)), np.unique(normal_train_unique)))
    print("Event types total in normal testing              : {}, including : {}".format(len(np.unique(normal_test_unique)), np.unique(normal_test_unique)))
    inter_abnormal_normal = []
    inter_train_abnormal = []
    inter_train_test = []
    for abnormal_seq in abnormal_unique:
        for normal_seq in normal_unique:
            if np.array_equal(abnormal_seq,normal_seq):
                inter_abnormal_normal.append(abnormal_seq)
                break
        for train_seq in normal_train_unique:
            if np.array_equal(abnormal_seq,train_seq):
                inter_train_abnormal.append(abnormal_seq)
                break
    print("Number of sequences exists in both normal and abnormal : {}".format(len(inter_abnormal_normal)))
    print("Number of sequences exists in both train and abnormal : {}".format(len(inter_train_abnormal)))
    for test_normal in normal_test_unique:
        for normal_seq in normal_unique:
            if np.array_equal(test_normal, normal_seq):
                print("{} is equel {}".format(test_normal, normal_seq))
                inter_train_test.append(test_normal)
                break
    print("Number of sequences exists in both train and test normal: {}".format(len(inter_train_test)))

    
def do_sample():
    # Sample log sequnces genrated by aminer-deep, it sample based on the group, in this case the sample ratio is 0.1
    logs, lables, groups, results = load_raw_data(options['data_save_dir'] +  options['data_file_name'])
    train_logs, train_labels, train_groups, train_result, test_logs, test_labels, test_groups, test_result = data_sp_sampling(logs, lables, groups, results, 0.1)
    print("Writing Test Sequences in test_{}".format(options['data_file_name']))
    for i in range(len(test_labels)):
        singel_line = [test_groups[i], test_logs["Sequentials"][i], test_labels[i], test_result[i]]
        write_sequence(singel_line, "test_" + options['data_file_name'])
    
    print("Writing Traning Sequences in train_{}".format(options['data_file_name']))
    for i in range(len(train_labels)):
        singel_line = [train_groups[i], train_logs["Sequentials"][i], train_labels[i], train_result[i]]
        write_sequence(singel_line, "train_" + options['data_file_name'])

def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()

def write_sequence(data, filename):
    with open(options["data_save_dir"] + filename,'a') as seq_file:
        seq_file.write("{}: {}: {}: {}\n".format(data[0], data[1], data[2], data[3]))
    

def start_detector(options, io_mode, filename=None):
    if not options["learn_mode"]:
        Model = deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            num_keys=options["num_classes"],
        )
        dectector = Detector(Model, options)
        l_model = dectector.load_model()
    context = zmq.Context()
    zmq_pub_socket = context.socket(zmq.PUB)
    zmq_pub_socket.connect(options["zmq_sub_endpoint"])
    zmq_sub_socket = context.socket(zmq.SUB)
    zmq_sub_socket.connect(options["zmq_pub_endpoint"])
    zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, options["zmq_aminer_top"])
    
    while True:
        # waite for the aminer parsed log as sequence, the expected AMiner output is {topic}:{group_by_value}:{json_seq}
        print("Waiting for the AMiner output.......")
        msg = zmq_sub_socket.recv_string()
        print(msg)
        # split the oueput and clean it to extract the sequence
        top, group, seq = msg.split(":")
        seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
        # parse the sequence values ino integer values 
        seq = [int(x) for x in seq]
        # compose group, sequence and detector result into one line, the defult value for detector result is false
        result_line = [group, seq[:-1],seq[-1] ,False]
        
        # if the leaning mode is True escape detection part, and pass AMiner output to io mode
        if not options["learn_mode"]:
            result = dectector.detect_anomaly(l_model, seq[:-1], seq[-1])
            result_line[3] = result
            print("Sending the detector result: {}".format(result_line))
            zmq_pub_socket.send_string("{}:{}".format(options["zmq_detector_top"], json.dumps(result_line)))
        if io_mode:
            print("Writing aminer & detector output in {}".format(filename))
            write_sequence(result_line,filename)

def predict(options):
    logs, lables, groups, results = load_raw_data(options['data_save_dir'] +  options['data_file_name'])
    Model = deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            num_keys=options["num_classes"],
        )
    dectector = Detector(Model, options)
    l_model = dectector.load_model()
    tp = 0
    fn = 0
    total = 0
    for i in range(len(lables)):
        total += 1
        result = dectector.detect_anomaly(l_model, logs["Sequentials"][i], lables[i])
        if result:
            tp += 1
        else:
            fn += 1
        print("{} - {} - {} - {}".format(groups[i],lables[i],logs["Sequentials"][i],result))
    print("Total Events: {}, True:{}, False:{}".format(total,tp,fn))


if __name__ == "__main__":
    filename = time.strftime('%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['withio', 'without', 'train', 'sample', 'predict','analyze'])
    args = parser.parse_args()
    if args.mode == 'withio':
        start_detector(options,True, filename)
    elif args.mode == 'without':
        start_detector(options,False)
    elif args.mode == 'train':
        train()
    elif args.mode == 'sample':
        do_sample()
    elif args.mode == 'predict':
        predict(options)
    elif args.mode == 'analyze':
        do_analyze()
    else:
        print('Invalid input')
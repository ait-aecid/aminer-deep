"""
Overall configration file, used by the detector_launcher.py and zmqproxy.py
"""
options = dict()

# data configration data_save_dir is dir where the logs will be stored if io mode is True
options["data_save_dir"] = "/home/ubuntu/aminer-deep/data/"
# the file will be used for tranning
options['data_file_name'] = "Ex03_dnsmask/125009"
options["device"] = "cpu"

# currently support one feature, sequentials
options['sequentials'] = True

# Model
options["input_size"] = 1
options["hidden_size"] = 64
options["num_layers"] = 2
options["num_classes"] = 10

# Train
options["batch_size"] = 2048
options["accumulation_step"] = 1
options["optimizer"] = "adam"
options["lr"] = 0.001
options["max_epoch"] = 100
options["lr_step"] = (300, 350)
options["lr_decay_ratio"] = 0.1
options["resume_path"] = None
options["model_name"] = "dnsmask"
options["save_dir"] = "/home/ubuntu/aminer-deep/result/aminer-deep/ex03-dns/"

# Detector
options[
    "model_path"
] = "/home/ubuntu/aminer-deep/result/aminer-deep/ex03-dns/dnsmask_last.pth"
options["num_candidates"] = 1


# ZMQ configration, the endpoint is presented from proxy point of view
options["zmq_pub_endpoint"] = "tcp://127.0.0.1:5559"
options["zmq_sub_endpoint"] = "tcp://127.0.0.1:5560"
options["zmq_aminer_top"] = "aminer"
options["zmq_detector_top"] = "deep-aminer"
options["learn_mode"] = True

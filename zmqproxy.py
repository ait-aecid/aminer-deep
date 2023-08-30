
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import zmq
from config import *


def main():

    context = zmq.Context()

    # Socket facing producers
    frontend = context.socket(zmq.XPUB)
    frontend.bind(options["zmq_pub_endpoint"])

    # Socket facing consumers
    backend = context.socket(zmq.XSUB)
    backend.bind(options["zmq_sub_endpoint"])

    zmq.proxy(frontend, backend)

    # 
    frontend.close()
    backend.close()
    context.term()

if __name__ == "__main__":
    main()
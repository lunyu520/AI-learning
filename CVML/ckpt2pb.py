from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph('./checkpoint/mnist.pbtxt', "", False,
                          './model/ckpt', "",
                          "", "",
                          './checkpoint/model.pb', True, ""
                          )

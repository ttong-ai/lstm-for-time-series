import numpy as np
import random
import string
import tensorflow as tf
from tensorflow.python.client import device_lib
from IPython.display import display, HTML
import logging

logger = logging.getLogger('gpu_compute')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('gpu_compute.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s::%(name)s::%(levelname)s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def reset_graph(seed=42):
    """Reset default tensorflow graph"""
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>" % size
    return strip_def


def show_graph(graph_def=None, max_const_size=32):
    """Visualize TensorFlow graph within the notebook"""
    if graph_def is None:
        graph_def = tf.get_default_graph()
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))
    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


class LSTM(object):
    """Container for multi-time-step and multi-layered LSTM framework"""

    def __init__(self, batch_size=None, n_input_features=None, n_states=50, n_layers=1, n_time_steps=10,
                 activation=tf.nn.relu, keep_prob=0.5, l1_reg=1e-2, l2_reg=1e-3,
                 start_learning_rate=0.001, decay_steps=1, decay_rate=0.3,
                 iter_per_id=10, forward_step=1):
        self.n_input_features = n_input_features
        self.batch_size = batch_size
        self.n_states = n_states
        self.n_layers = n_layers
        self.n_time_steps = n_time_steps
        self.activation = activation
        self.keep_prob = keep_prob
        self.l1_reg_scale = l1_reg
        self.l2_reg_scale = l2_reg
        self.start_learning_rate = start_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.iter_per_id = iter_per_id
        self.forward_step = forward_step
        self.sessid = None
        self.device_list = None
        self.graph = tf.Graph()
        self.graph_keys = None

        # Locate available computing devices and save in self.device_list
        self.device_list = self.find_compute_devices()
        try:
            self.compute_device = self.device_list['gpu'][0]
        except (KeyError, IndexError) as msg:
            print(msg)
            self.compute_device = self.device_list['cpu'][0]  # default to cpu as computing device

    def __call__(self):
        pass

    def logging_session_para(self):
        logger.info(f"[{self.sessid}] Session start")
        logger.info(f"[{self.sessid}] Input features: {self.n_input_features}")
        logger.info(f"[{self.sessid}] Num of units in each LSTM cell: {self.n_states}")
        logger.info(f"[{self.sessid}] Num of stacked LSTM layers: {self.n_layers}")
        logger.info(f"[{self.sessid}] Num of unrolled time steps: {self.n_time_steps}")
        logger.info(f"[{self.sessid}] Activation function: {self.activation.__name__}")
        logger.info(f"[{self.sessid}] Dropout rate during training: {1 - self.keep_prob}")
        logger.info(f"[{self.sessid}] L1 regularization: {self.l1_reg_scale}")
        logger.info(f"[{self.sessid}] L2 regularization: {self.l2_reg_scale}")
        logger.info(f"[{self.sessid}] Start learning rate: {self.start_learning_rate}")
        logger.info(f"[{self.sessid}] Learning rate decay steps: {self.decay_steps}")
        logger.info(f"[{self.sessid}] Learning rate decay rate: {self.decay_rate}")
        logger.info(f"[{self.sessid}] Inner iteration per id: {self.iter_per_id}")
        logger.info(f"[{self.sessid}] Forward prediction period: {self.forward_step}")

    @staticmethod
    def find_compute_devices():
        device_list = device_lib.list_local_devices()
        gpu, cpu = [], []
        for device in device_list:
            if device.name.find('GPU') != -1:
                gpu.append(device.name)
            if device.name.find('CPU') != -1:
                cpu.append(device.name)
        assert len(cpu) >= 1  # assert at least cpu resource is available
        return dict({'gpu': gpu, 'cpu': cpu})

    def show_compute_devices(self):
        if self.compute_device is None:
            self.device_list = self.find_compute_devices()
        print("Following compute devices available\n  ", self.device_list)

    def set_compute_device(self, type='gpu', seq=0):
        try:
            self.compute_device = self.device_list[type][seq]
        except (KeyError, IndexError) as msg:
            print("Error in selecting target device, defaulting to CPU as compute device. \n"
                  "Please use show_compute_devices() to list available compute devices.")
            self.compute_device = self.device_list['cpu'][0]  # default to cpu as computing device

    def reset_graph(self):
        del self.graph
        self.graph = tf.Graph()

    def show_graph(self, max_const_size=32):
        show_graph(self.graph.as_graph_def(), max_const_size)

    @staticmethod
    def get_tf_normal_variable(shape, mean=0.0, stddev=0.6):
        return tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=stddev), validate_shape=False)

    def create_lstm_graph(self, n_input_features=None, reset_graph=True, verbose=1):
        """Build the Tensorflow based LSTM network
        Input::
        n_input_features: number of input features, there is no default value and has to be provided.
        Return::  tensor references that need to be referenced later
        """
        if n_input_features is None:
            assert self.n_input_features is not None
        else:
            self.n_input_features = n_input_features

        if reset_graph:
            print("Warning: current graph if defined will be lost. ")
            self.reset_graph()

        # Build the network on the specified compute device
        with tf.device(self.compute_device):
            with self.graph.as_default():
                # Define input placeholder X
                with tf.name_scope('input'):
                    with tf.name_scope('X'):
                        # [None, n_time_steps, n_input_features]
                        X = tf.placeholder(tf.float32, shape=[self.batch_size, self.n_time_steps,
                                                              self.n_input_features])

                with tf.name_scope('hyperparameters'):
                    with tf.name_scope('keep_prob'):  # Define keep_prob placeholder for dropout
                        keep_prob = tf.placeholder(tf.float32)
                    with tf.name_scope('in_sample_cutoff'):  # split point between training and test
                        in_sample_cutoff = tf.placeholder(tf.int32, shape=(), name='in_sample_cutoff')

                # Define multilayer LSTM network
                with tf.name_scope('model'):
                    with tf.name_scope('rnn'):
                        # Adding dropout wrapper layer and LSTM cells with the number of hidden units
                        # in each LSTMCell as n_states.
                        lstm_layers = [
                            tf.nn.rnn_cell.DropoutWrapper(
                                tf.nn.rnn_cell.LSTMCell(num_units=self.n_states, use_peepholes=False,
                                                        forget_bias=1.0, activation=self.activation,
                                                        state_is_tuple=True), output_keep_prob=self.keep_prob)
                            for _ in range(self.n_layers)
                        ]
                        multilayer_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers, state_is_tuple=True)

                    with tf.name_scope('dynamical_unrolling'):
                        # init_states = []
                        # for _ in range(len(lstm_layers)):
                        #     cell_state = get_tf_normal_variable((batch_size, n_states))
                        #     hidden_state = get_tf_normal_variable((batch_size, n_states))
                        #     state_tuple = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
                        #     init_states.append(state_tuple)

                        # Use dynamic_rnn to dynamically unroll the time steps when doing the computation
                        # outputs contain the output from all the time steps, so it should have
                        # shape [batch_size, n_time_steps, n_states]
                        # When time_major is set to True, the outputs shape should be [n_time_steps, batch_size, n_states]
                        # states contain the all the internal states at the last time step.  It is a tuple with elements
                        # corresponding to n_layers. Each tuple element itself is a LSTMStateTuple with c and h tensors.
                        outputs, states = tf.nn.dynamic_rnn(cell=multilayer_cell, inputs=X,
                                                            initial_state=None, dtype=tf.float32,  # tuple(init_states)
                                                            swap_memory=True, time_major=False)

                    # Use a fully-connected layer to convert the multi-state vector into a single scalar representing
                    # the variable to be predicted
                    with tf.name_scope('fc'):
                        with tf.name_scope('W'):
                            W_fc1 = self.get_tf_normal_variable([self.n_states, 1])
                        with tf.name_scope('b'):
                            b_fc1 = self.get_tf_normal_variable([1])
                        with tf.name_scope('pred'):
                            pred = tf.matmul(states[-1][1],
                                             W_fc1) + b_fc1  # states[-1][1] is the h states of the last layer cell

                # Placeholder for the output (label)
                with tf.name_scope('label'):
                    y = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='y_label')
                    # this is important - we only want to train on the in-sample set of rows using TensorFlow
                    y_is = y[0:in_sample_cutoff]
                    pred_is = pred[0:in_sample_cutoff]
                    # also extract out of sample predictions and actual values,
                    # we'll use them for evaluation while training the model.
                    y_oos = y[in_sample_cutoff:]
                    pred_oos = pred[in_sample_cutoff:]

                with tf.name_scope('stats'):
                    # Pearson correlation to evaluate the model, all using in-sample training data
                    covariance = tf.reduce_sum(
                        tf.matmul(
                            tf.transpose(tf.subtract(pred_is, tf.reduce_mean(pred_is))),
                            tf.subtract(y_is, tf.reduce_mean(y_is))
                        )
                    )
                    var_pred = tf.reduce_sum(
                        tf.square(tf.subtract(pred_is, tf.reduce_mean(pred_is)))
                    )
                    var_y = tf.reduce_sum(tf.square(tf.subtract(y_is, tf.reduce_mean(y_is))))
                    pearson_corr = covariance / tf.sqrt(var_pred * var_y)

                with tf.name_scope('hyperparameters'):
                    # set up adaptive learning rate:
                    # Ratio of global_step / decay_steps is designed to indicate how far we've progressed in training.
                    # the ratio is 0 at the beginning of training and is 1 at the end.
                    global_step = tf.placeholder(tf.float32)

                    # tf.train.exponetial_decay is calculated as:
                    #     decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
                    # adaptive_learning_rate will thus change from the starting learningRate to learningRate * decay_rate
                    # in order to simplify the code, we are fixing the total number of decay steps at 1 and pass global_step
                    # as a fraction that starts with 0 and tends to 1.
                    adaptive_learning_rate = tf.train.exponential_decay(
                        learning_rate=self.start_learning_rate,  # Start with this learning rate
                        global_step=global_step,  # global_step / total_steps shows how far we've progressed in training
                        decay_steps=self.decay_steps,
                        decay_rate=self.decay_rate
                    )

                # Define loss and optimizer
                # Note the loss only involves in-sample rows
                # Regularization is added in the loss function to avoid over-fitting
                rnn_variables = lstm_variables = [v for v in tf.trainable_variables()
                                                  if v.name.startswith('rnn')]

                with tf.name_scope('loss'):
                    loss = tf.nn.l2_loss(tf.subtract(y_is, pred_is)) + \
                           tf.contrib.layers.apply_regularization(
                               tf.contrib.layers.l1_l2_regularizer(scale_l1=self.l1_reg_scale, scale_l2=self.l2_reg_scale),
                               tf.trainable_variables())

                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=adaptive_learning_rate).minimize(loss)

                with tf.name_scope('summary'):
                    tf.summary.scalar("pearson_corr", pearson_corr)
                    tf.summary.scalar("loss", loss)
                    summary_op = tf.summary.merge_all()

                # Write the graph to summary
                writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())

                self.graph_keys = dict(X=X,
                                       y=y,
                                       y_oos=y_oos,
                                       pred=pred,
                                       pred_oos=pred_oos,
                                       keep_prob=keep_prob,
                                       in_sample_cutoff=in_sample_cutoff,
                                       global_step=global_step,
                                       states=states,
                                       outputs=outputs,
                                       loss=loss,
                                       optimizer=optimizer,
                                       pearson_corr=pearson_corr,
                                       adaptive_learning_rate=adaptive_learning_rate,
                                       summary_op=summary_op,
                                       writer=writer
                                       )

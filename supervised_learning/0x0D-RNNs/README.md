# RECURRENT NEURAL NETWORKS

### Tasks

#### [RNN Cell](./0-rnn_cell.py)
- Create the class RNNCell that represents a cell of a simple RNN

#### [RNN](./1-rnn.py)
- Write the function def rnn(rnn_cell, X, h_0): that performs forward propagation for a simple RNN

#### [GRU Cell](./2-gru_cell.py)
- Create the class GRUCell that represents a gated recurrent unit

#### [LSTM Cell](./3-lstm_cell.py)
- Create the class LSTMCell that represents an LSTM unit

#### [Deep RNN](./4-deep_rnn.py)
- Write the function def deep_rnn(rnn_cells, X, h_0): that performs forward propagation for a deep RNN

#### [Bidirectional Cell Forward](./5-bi_forward.py)
- Create the class BidirectionalCell that represents a bidirectional cell of an RNN

#### [Bidirectional Cell Backward](./6-bi_backward.py)
- Update the class BidirectionalCell, based on 5-bi_forward.py:
    - public instance method def backward(self, h_next, x_t): that calculates the hidden state in the backward direction for one time step

#### [Bidirectional Output](./7-bi_output.py)
- Update the class BidirectionalCell, based on 6-bi_backward.py
    - public instance method def output(self, H): that calculates all outputs for the RNN

#### [Bidirectional RNN](./8-bi_rnn.py)
- Write the function def bi_rnn(bi_cell, X, h_0, h_t): that performs forward propagation for a bidirectional RNN

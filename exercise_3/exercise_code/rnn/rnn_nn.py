import torch
import torch.nn as nn
import torch.nn.functional as fn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        super(RNN, self).__init__()
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_dense = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.state_dense = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

        self.h = torch.randn(hidden_size, hidden_size)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        seq_len, batch_size, input_size = x.shape

        h = h or torch.zeros(size=(1, batch_size, self.hidden_size))
        h_seq = torch.zeros(size=x.shape)

        for i, seq in enumerate(x):
            h = torch.tanh(self.state_dense(h) + self.input_dense(seq))
            h_seq[i] = h
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq, h
    
    
class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        super(LSTM, self).__init__()

        self.input_f = nn.Linear(input_size, hidden_size, bias=True)
        self.input_i = nn.Linear(input_size, hidden_size, bias=True)
        self.input_o = nn.Linear(input_size, hidden_size, bias=True)
        self.input_c = nn.Linear(input_size, hidden_size, bias=True)

        self.state_f = nn.Linear(hidden_size, hidden_size, bias=True)
        self.state_i = nn.Linear(hidden_size, hidden_size, bias=True)
        self.state_o = nn.Linear(hidden_size, hidden_size, bias=True)
        self.state_c = nn.Linear(hidden_size, hidden_size, bias=True)

        self.hidden_size = hidden_size
       
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        seq_len, batch_size, input_size = x.shape

        h = h or torch.zeros((1, batch_size, self.hidden_size))
        c = c or torch.zeros((1, batch_size, self.hidden_size))
        h_seq = torch.zeros(x.shape)

        for i, seq in enumerate(x):
            # Forget gate
            f_t = torch.sigmoid(self.input_f(seq) + self.state_f(h))

            # Input gate
            i_t = torch.sigmoid(self.input_i(seq) + self.state_i(h))

            # Output gate
            o_t = torch.sigmoid(self.input_o(seq) + self.state_o(h))

            g_t = torch.tanh(self.input_c(seq) + self.state_c(h))

            # Memory cell
            c = f_t * c + i_t * g_t

            # State
            h = o_t * torch.tanh(c)

            h_seq[i] = h

        return h_seq, (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        pass
       
    def forward(self, x):
        pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = classes

        self.lstm = nn.LSTM(input_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, classes)
    
    def forward(self, x):
        # Set initial hidden and cell states
        h = torch.zeros(1, x.shape[1], self.hidden_size)
        c = torch.zeros(1, x.shape[1], self.hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h, c))
        out = self.linear(out[-1].view(x.shape[1], -1))

        return out

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


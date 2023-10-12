import torch
import torch.nn as nn


class ParametricLSTMCNN(nn.Module):
    def __init__(self, num_layers_conv, num_kernels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, inputlstm, batch_size):
        super(ParametricLSTMCNN, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.padding_sizes = padding_sizes
        self.hidden_size_lstm = hidden_size_lstm
        self.num_layers_conv = num_layers_conv
        self.num_layer_lstm = num_layers_lstm
        self.hidden_neurons_dense = hidden_neurons_dense
        self.seq = seq
        self.inputlstm = inputlstm
        self.output_shape = []
        self.num_kernels = num_kernels # number of kernels in each layer
        self.batch_size = batch_size
        self.outputs = [self.seq]
        
        def output_shape_calc(i):
        
            output_height = (self.outputs[i-1] - self.kernel_sizes[i] + 2*self.padding_sizes[i])/self.stride_sizes[i] + 1
            return output_height
        
        for i in range(1, self.num_layers_conv+1):
            
            self.outputs.append(int(output_shape_calc(i-1)))
            if self.outputs[i-1] <= 0: 
                print('change inputs')
                break
            
            if i == 1:
                
                self.conv1 = nn.Conv1d(in_channels=self.seq, out_channels=self.num_kernels[i-1], kernel_size=self.kernel_sizes[i-1], stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1])
                self.batch1 = nn.BatchNorm1d(self.num_kernels[i-1])
                
            elif i == self.num_layers_conv:
                setattr(self, 'conv'+str(i), nn.Conv1d(in_channels = self.num_kernels[i-2], out_channels=self.num_layer_lstm, kernel_size=int(self.kernel_sizes[i-1]), stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1]))
                setattr(self, 'batch'+str(i), nn.BatchNorm1d(int(self.num_layer_lstm)))
                
            else:
                setattr(self, 'conv'+str(i), nn.Conv1d(in_channels=self.num_kernels[i-2], out_channels=self.num_kernels[i-1], kernel_size=int(self.kernel_sizes[i-1]), stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1]))
                setattr(self, 'batch'+str(i), nn.BatchNorm1d(self.num_kernels[i-1]))
    
            # first layer must be lstm
        self.lstm = nn.LSTM(input_size=self.inputlstm, hidden_size=self.hidden_size_lstm, num_layers=self.num_layer_lstm, batch_first=True, dropout=0.2)  # changed the input becasue the data from physical model is different

        # linear layer
        self.dense2 = nn.Linear(self.hidden_size_lstm, 1)
    
        self.relu = nn.LeakyReLU(0.05)
        

    def hyperparameter_check(self):
        x = 0
        for i in self.output_shape:
            if i == int(i):
                x += 1
        if x == len(self.output_shape):
            return True
        else:
            return False

    def weights_init(self):
        nn.init.xavier_normal_(self.lstm.weight)

        for i in range(self.num_layers_conv):
            conv_name = f'conv{i+1}'
            conv_layer = getattr(self, conv_name)
            nn.init.xavier_normal_(conv_layer.weight)

    def forward(self, x, verbose=False):
        if verbose:
            print(f'shape of x is {x.shape}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        out = x
        
        
        for i in range(self.num_layers_conv):
            conv_name = f'conv{i+1}'  # or use whatever naming scheme you used for your conv layers
            
            conv_layer = getattr(self, conv_name)
            out = self.relu(conv_layer(out))
            
            if verbose:
                print(f'shape after conv layer {i+1} is {out.shape}')
            batch_name = f'batch{i+1}'
            batch_norm = getattr(self, batch_name)
            out = batch_norm(out)
            if verbose:
                print(f'shape after batch layer {i+1} is {out.shape}')


        h_0 = torch.randn(self.num_layer_lstm, out.shape[0], self.hidden_size_lstm).to(device).float()
        c_0 = torch.randn(self.num_layer_lstm, out.shape[0], self.hidden_size_lstm).to(device).float()

        # output of lstm
        output, (hn, cn) = self.lstm(out, (h_0, c_0))  # lstm with input, hidden, and internal state
        if verbose:
            print(f'shape after lstm is {output.shape}')
        # output of first dense layer

        out = self.relu(self.dense2(output))
        if verbose:
            print(f'shape after second dense layer is {out.shape}')

        if verbose:
            print(f'output shape is {out.shape}')

        return out

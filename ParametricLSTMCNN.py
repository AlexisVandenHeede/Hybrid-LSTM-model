import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


class ParametricLSTMCNN(nn.Module):
    def __init__(self, num_layers_conv, output_channels, kernel_sizes, stride_sizes, padding_sizes, hidden_size_lstm, num_layers_lstm, hidden_neurons_dense, seq, inputlstm):
        super(ParametricLSTMCNN, self).__init__()
        self.output_channels = output_channels
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
        for i in range(self.num_layers_conv):
            if i == 0:
                if self.kernel_sizes[i] > self.hidden_neurons_dense[i] + 2 * self.padding_sizes[i]:
                    print('changed kernel size')
                    self.kernel_sizes[i] = self.hidden_neurons_dense[i] + 2 * self.padding_sizes[i] - 1
                    output_shape_1 = (self.hidden_neurons_dense[i] - self.kernel_sizes[i] + 2 * self.padding_sizes[i])/self.stride_sizes[i] + 1
                    self.output_shape.append(output_shape_1)
                else:
                    output_shape_1 = (self.hidden_neurons_dense[i] - self.kernel_sizes[i] + 2 * self.padding_sizes[i])/self.stride_sizes[i] + 1
                    self.output_shape.append(output_shape_1)
            else:
                if self.kernel_sizes[i] > self.output_shape[i-1] + 2 * self.padding_sizes[i-1]:
                    print('changed kernel size')
                    self.kernel_sizes[i] = self.output_shape[i-1] + 2 * self.padding_sizes[i-1] - 1
                    output_shape = (self.output_shape[i-1] - self.kernel_sizes[i] + 2 * self.padding_sizes[i])/self.stride_sizes[i] + 1
                    self.output_shape.append(output_shape)
                else:
                    output_shape = (self.output_shape[i-1] - self.kernel_sizes[i] + 2 * self.padding_sizes[i])/self.stride_sizes[i] + 1
                    self.output_shape.append(output_shape)

        # print(self.output_shape)
        if self.output_shape[-1] <= 0:
            print('change inputs')
        else:
            # first layer must be lstm
            self.lstm = nn.LSTM(self.inputlstm, self.hidden_size_lstm, num_layers=self.num_layer_lstm, batch_first=True, dropout=0.2)  # changed the input becasue the data from physical model is different

            # then dense
            self.dense1 = nn.Linear(self.hidden_size_lstm, self.hidden_neurons_dense[0])
            # set the conv and batchnorm layers
            for i in range(1, self.num_layers_conv+1):
                if i == 1:
                    if self.num_layers_conv == 1:
                        self.conv1 = nn.Conv1d(in_channels=self.seq, out_channels=1, kernel_size=self.kernel_sizes[i-1], stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1])
                        self.batch1 = nn.BatchNorm1d(1)
                    else:
                        self.conv1 = nn.Conv1d(in_channels=self.seq, out_channels=self.output_channels[i-1], kernel_size=self.kernel_sizes[i-1], stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1])
                        self.batch1 = nn.BatchNorm1d(self.output_channels[i-1])
                elif i == self.num_layers_conv:
                    setattr(self, 'conv'+str(i), nn.Conv1d(in_channels=self.output_channels[i-2], out_channels=1, kernel_size=int(self.kernel_sizes[i-1]), stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1]))
                    setattr(self, 'batch'+str(i), nn.BatchNorm1d(1))
                else:
                    setattr(self, 'conv'+str(i), nn.Conv1d(in_channels=int(self.output_channels[i-2]), out_channels=self.output_channels[i-1], kernel_size=int(self.kernel_sizes[i-1]), stride=self.stride_sizes[i-1], padding=self.padding_sizes[i-1]))
                    setattr(self, 'batch'+str(i), nn.BatchNorm1d(self.output_channels[i-1]))

            # dense layers after conv
            for i in range(2, len(self.hidden_neurons_dense)+1):
                if i == 2:
                    setattr(self, 'dense'+str(i), nn.Linear(int(self.output_shape[-1]), int(self.hidden_neurons_dense[1])))
                elif i == len(self.hidden_neurons_dense):
                    setattr(self, 'dense'+str(i), nn.Linear(in_features=self.hidden_neurons_dense[i-2], out_features=1))
                else:
                    setattr(self, 'dense'+str(i), nn.Linear(in_features=self.hidden_neurons_dense[i-2], out_features=self.hidden_neurons_dense[i-1]))

            self.relu = nn.LeakyReLU()
            self.dropout = nn.Dropout(0.2)

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
        seed_value = 0
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) 
        cudnn.deterministic = True
        cudnn.benchmark = False

        for i in range(1, len(self.hidden_neurons_dense)+1):
            dense_name = f'dense{i}'
            dense_layer = getattr(self, dense_name)
            nn.init.xavier_normal_(dense_layer.weight)

        for i in range(self.num_layers_conv):
            conv_name = f'conv{i+1}'
            conv_layer = getattr(self, conv_name)
            nn.init.xavier_normal_(conv_layer.weight)

    def forward(self, x, verbose=False):
        if verbose:
            print(f'shape of x is {x.shape}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h_0 = torch.randn(self.num_layer_lstm, x.shape[0], self.hidden_size_lstm).to(device).float()
        c_0 = torch.randn(self.num_layer_lstm, x.shape[0], self.hidden_size_lstm).to(device).float()

        # output of lstm
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        if verbose:
            print(f'shape after lstm is {output.shape}')
        # output of first dense layer
        out = self.relu(self.dense1(self.relu(output)))
        if verbose:
            print(f'shape after first dense layer is {out.shape}')

        for i in range(self.num_layers_conv):
            conv_name = f'conv{i+1}'  # or use whatever naming scheme you used for your conv layers
            conv_layer = getattr(self, conv_name)
            out = conv_layer(out)
            if verbose:
                print(f'shape after conv layer {i+1} is {out.shape}')
            batch_name = f'batch{i+1}'
            batch_norm = getattr(self, batch_name)
            out = batch_norm(out)
            out = self.relu(out)
            if verbose:
                print(f'shape after batch layer {i+1} is {out.shape}')

        for j in range(1, len(self.hidden_neurons_dense)-1):
            # print(F'moving on to dense')
            dense_name = f'dense{j+1}'
            dense_layer = getattr(self, dense_name)
            out = self.relu(dense_layer(out))
            if verbose:
                print(f'shape after dense layer {j+1} is {out.shape}')

        out = self.dropout(out)

        last_dense = f'dense{len(self.hidden_neurons_dense)}'
        last_dense_layer = getattr(self, last_dense)
        out = last_dense_layer(out)

        if verbose:
            print(f'output shape is {out.shape}')

        return out

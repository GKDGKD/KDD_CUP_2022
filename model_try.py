import torch
import torch.nn as nn

class SpatialTemporalModel(nn.Module):
    def __init__(self, num_nodes, input_seq_len, output_seq_len, num_features, tcn_channels, tcn_layers):
        super(SpatialTemporalModel, self).__init__()

        # CNN for spatial-temporal feature extraction
        self.cnn1 = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(num_features, 128, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # TCN for temporal prediction
        self.tcn = TemporalConvNet(num_nodes, 192, tcn_channels, tcn_layers)

        # Output layer
        self.fc = nn.Linear(tcn_channels, num_nodes * output_seq_len)

    def forward(self, x):
        # x shape: [batch_size, num_nodes, input_seq_len, num_features]

        # Apply multiple CNNs for spatial-temporal feature extraction
        cnn1_output = self.cnn1(x.permute(0, 3, 1, 2))  # Reshape for CNN
        cnn2_output = self.cnn2(x.permute(0, 3, 1, 2))  # Reshape for CNN
        cnn_output = torch.cat([cnn1_output, cnn2_output], dim=1)
        cnn_output = cnn_output.reshape(cnn_output.size(0), -1)  # Use reshape

        # Apply TCN for temporal prediction
        tcn_output = self.tcn(cnn_output)

        # Output layer
        predictions = self.fc(tcn_output)

        return predictions.view(-1, num_nodes, output_seq_len)

class TemporalConvNet(nn.Module):
    def __init__(self, num_nodes, input_size, num_channels, num_layers):
        super(TemporalConvNet, self).__init__()

        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels
            layers += [
                TemporalBlock(num_nodes, in_channels, num_channels, kernel_size=3, dilation=dilation_size),
                nn.ReLU()
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, num_nodes, input_size, output_size, kernel_size, dilation):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(num_nodes, input_size, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(num_nodes, input_size, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)

        self.downsample = nn.Conv1d(num_nodes, output_size, 1) if input_size != output_size else None

    def forward(self, x):
        y = torch.relu(self.conv1(x))
        y = torch.relu(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + y

# Example usage
num_nodes = 134
input_seq_len = 48
output_seq_len = 24
num_features = 10
tcn_channels = 64
tcn_layers = 2

model = SpatialTemporalModel(num_nodes, input_seq_len, output_seq_len, num_features, tcn_channels, tcn_layers)
input_data = torch.randn((32, num_nodes, input_seq_len, num_features))  # Example input
print(input_data.shape)
output = model(input_data)
print(output.shape)

from torch import nn


class SimpleConv(nn.Module):

    def __init__(self, mode: ["classification", "regression"] = "classification"):
        super(SimpleConv, self).__init__()
        self.mode = mode
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2), # -> 300*300*8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=10), # -> 30*30*8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # -> 30*30*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3), # -> 10*10*16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> 10*10*32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential( # for classification
            nn.Linear(3200, 1600),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1600, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 1),
        )
        # self.fc2 = nn.Sequential( # for regression
        #    nn.ReLU(),
        #    nn.Dropout(),
        #    nn.Linear(500, 1),
        # )

        self.stem = nn.Sequential(
            self.layer1,
            nn.Dropout(),
            self.fc,
        )
        if self.mode == "classification":
            self.classification = nn.Sequential(self.stem, nn.Sigmoid())
        else:
            self.regression = self.stem

    def forward(self, x):
        if self.mode == "classification":
            x = self.classification(x)
        else:
            x = self.regression(x)
        return x
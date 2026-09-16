import torch
import torch.nn as nn


# ============================================================
# CIFAR-10 RESNET34
# ============================================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out += identity

        out = self.relu(out)

        return out


# ============================================================
# RESNET34
# ============================================================

class ResNet34(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()

        # ----------------------------------------------------
        # CIFAR-specific stem
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        # ----------------------------------------------------
        # ResNet34 layers
        #
        # ResNet34 = 3, 4, 6, 3 blocks
        # ----------------------------------------------------

        self.layer1 = self._make_layer(
            64,
            64,
            blocks=3,
            stride=1
        )

        self.layer2 = self._make_layer(
            64,
            128,
            blocks=4,
            stride=2
        )

        self.layer3 = self._make_layer(
            128,
            256,
            blocks=6,
            stride=2
        )

        self.layer4 = self._make_layer(
            256,
            512,
            blocks=3,
            stride=2
        )

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            512,
            num_classes
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # MAKE RESNET LAYER
    # ========================================================

    def _make_layer(
        self,
        in_channels,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        # First block may downsample
        layers.append(
            BasicBlock(
                in_channels,
                out_channels,
                stride
            )
        )

        # Remaining blocks
        for _ in range(1, blocks):

            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)

        # Flatten
        x = torch.flatten(
            x,
            1
        )

        # Classifier
        x = self.fc(x)

        return x

    # ========================================================
    # INITIALIZE WEIGHTS
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.constant_(
                    module.weight,
                    1
                )

                nn.init.constant_(
                    module.bias,
                    0
                )

            elif isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    0,
                    0.01
                )

                nn.init.constant_(
                    module.bias,
                    0
                )


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Create ResNet34 from scratch
    model = ResNet34(
        num_classes=10
    ).to(device)

    print("\n========== MODEL ==========")

    print(model)

    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters    :", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    # --------------------------------------------------------
    # Test forward pass
    # --------------------------------------------------------

    dummy_input = torch.randn(
        4,
        3,
        32,
        32
    ).to(device)

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print(
        "\nInput shape :",
        dummy_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print("==========================")
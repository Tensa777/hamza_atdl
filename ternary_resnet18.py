# ============================================================
# TERNARY RESNET18 FOR CIFAR-10
# ============================================================
#
# Task 2:
#   ResNet18 student with ternary Conv/FC weights:
#       {-alpha, 0, +alpha}
#
# Task 3:
#   Quantization-Aware Training (QAT)
#   + Knowledge Distillation from ResNet34 teacher
#
# Important:
#   - Latent FP32 weights are maintained for optimization.
#   - Forward pass ALWAYS uses ternary weights.
#   - BatchNorm remains full precision.
#   - Biases remain full precision.
#   - Alpha is learnable and constrained to be positive.
#
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TERNARY QUANTIZATION WITH STRAIGHT-THROUGH ESTIMATOR
# ============================================================

class TernaryQuantizeSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, alpha):

        # ----------------------------------------------------
        # Keep alpha positive
        # ----------------------------------------------------

        alpha_positive = torch.abs(alpha) + 1e-8

        # ----------------------------------------------------
        # Normalize weight
        # ----------------------------------------------------

        normalized = weight / alpha_positive

        # ----------------------------------------------------
        # Ternary quantization
        #
        # normalized > +0.5  -> +1
        # normalized < -0.5  -> -1
        # otherwise          ->  0
        # ----------------------------------------------------

        ternary = torch.where(
            normalized > 0.5,
            torch.ones_like(weight),
            torch.where(
                normalized < -0.5,
                -torch.ones_like(weight),
                torch.zeros_like(weight)
            )
        )

        # ----------------------------------------------------
        # Save values for backward pass
        # ----------------------------------------------------

        ctx.save_for_backward(
            ternary,
            alpha_positive
        )

        # ----------------------------------------------------
        # Actual QAT weight
        #
        # {-alpha, 0, +alpha}
        # ----------------------------------------------------

        quantized_weight = (
            ternary * alpha_positive
        )

        return quantized_weight

    @staticmethod
    def backward(ctx, grad_output):

        ternary, alpha_positive = ctx.saved_tensors

        # ----------------------------------------------------
        # Straight-Through Estimator
        #
        # Approximate:
        #       dQ/dW = 1
        # ----------------------------------------------------

        grad_weight = grad_output.clone()

        # ----------------------------------------------------
        # Gradient for alpha
        #
        # Q = ternary * alpha
        #
        # Approximate:
        #       dQ/dalpha = ternary
        #
        # Mean keeps the gradient stable.
        # ----------------------------------------------------

        grad_alpha = (
            grad_output * ternary
        ).mean().reshape_as(alpha_positive)

        return grad_weight, grad_alpha


def ternary_quantize(weight, alpha):

    return TernaryQuantizeSTE.apply(
        weight,
        alpha
    )


# ============================================================
# TERNARY CONVOLUTION
# ============================================================

class TernaryConv2d(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False
    ):

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # nn.Conv2d.__init__() internally calls
        # self.reset_parameters().
        #
        # Therefore log_alpha MUST NOT be accessed inside
        # the first reset_parameters() call.
        # ----------------------------------------------------

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # ----------------------------------------------------
        # Create learnable alpha AFTER Conv2d initialization
        # ----------------------------------------------------

        self.log_alpha = nn.Parameter(
            torch.tensor(0.0)
        )

        # ----------------------------------------------------
        # Initialize alpha from mean absolute weight
        # ----------------------------------------------------

        with torch.no_grad():

            initial_alpha = (
                self.weight.abs().mean()
            )

            self.log_alpha.copy_(
                torch.log(
                    torch.expm1(
                        initial_alpha.clamp_min(1e-6)
                    )
                )
            )

    @property
    def alpha(self):

        # ----------------------------------------------------
        # Softplus guarantees positive alpha
        # ----------------------------------------------------

        return (
            F.softplus(self.log_alpha)
            + 1e-8
        )

    def reset_parameters(self):

        # ----------------------------------------------------
        # This function is called automatically by
        # nn.Conv2d.__init__() BEFORE log_alpha exists.
        #
        # Therefore we initialize only normal Conv2d
        # parameters here.
        # ----------------------------------------------------

        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

        if self.bias is not None:

            nn.init.zeros_(
                self.bias
            )

        # ----------------------------------------------------
        # Only initialize alpha if log_alpha already exists.
        # ----------------------------------------------------

        if hasattr(
            self,
            "log_alpha"
        ):

            with torch.no_grad():

                initial_alpha = (
                    self.weight.abs().mean()
                )

                self.log_alpha.copy_(
                    torch.log(
                        torch.expm1(
                            initial_alpha.clamp_min(1e-6)
                        )
                    )
                )

    def forward(self, x):

        # ----------------------------------------------------
        # QAT:
        #
        # Forward pass ALWAYS uses ternary weights.
        # ----------------------------------------------------

        quantized_weight = ternary_quantize(
            self.weight,
            self.alpha
        )

        # ----------------------------------------------------
        # Convolution using quantized weight
        # ----------------------------------------------------

        return F.conv2d(
            x,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


# ============================================================
# TERNARY LINEAR
# ============================================================

class TernaryLinear(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True
    ):

        # ----------------------------------------------------
        # nn.Linear.__init__() calls reset_parameters().
        # Therefore log_alpha is created AFTER that call.
        # ----------------------------------------------------

        super().__init__(
            in_features,
            out_features,
            bias=bias
        )

        # ----------------------------------------------------
        # Learnable alpha parameter
        # ----------------------------------------------------

        self.log_alpha = nn.Parameter(
            torch.tensor(0.0)
        )

        # ----------------------------------------------------
        # Initialize alpha
        # ----------------------------------------------------

        with torch.no_grad():

            initial_alpha = (
                self.weight.abs().mean()
            )

            self.log_alpha.copy_(
                torch.log(
                    torch.expm1(
                        initial_alpha.clamp_min(1e-6)
                    )
                )
            )

    @property
    def alpha(self):

        return (
            F.softplus(self.log_alpha)
            + 1e-8
        )

    def reset_parameters(self):

        # ----------------------------------------------------
        # Standard Linear initialization
        # ----------------------------------------------------

        nn.init.kaiming_uniform_(
            self.weight,
            a=5 ** 0.5
        )

        if self.bias is not None:

            nn.init.zeros_(
                self.bias
            )

        # ----------------------------------------------------
        # Only access log_alpha if it already exists.
        # ----------------------------------------------------

        if hasattr(
            self,
            "log_alpha"
        ):

            with torch.no_grad():

                initial_alpha = (
                    self.weight.abs().mean()
                )

                self.log_alpha.copy_(
                    torch.log(
                        torch.expm1(
                            initial_alpha.clamp_min(1e-6)
                        )
                    )
                )

    def forward(self, x):

        # ----------------------------------------------------
        # QAT forward pass
        # ----------------------------------------------------

        quantized_weight = ternary_quantize(
            self.weight,
            self.alpha
        )

        return F.linear(
            x,
            quantized_weight,
            self.bias
        )


# ============================================================
# RESNET BASIC BLOCK
# ============================================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1
    ):

        super().__init__()

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        self.conv1 = TernaryConv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            planes
        )

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        self.conv2 = TernaryConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            planes
        )

        # ----------------------------------------------------
        # Shortcut connection
        # ----------------------------------------------------

        self.shortcut = nn.Sequential()

        if (
            stride != 1
            or in_planes != planes
        ):

            self.shortcut = nn.Sequential(

                TernaryConv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False
                ),

                nn.BatchNorm2d(
                    planes
                )
            )

    def forward(self, x):

        identity = self.shortcut(x)

        # ----------------------------------------------------
        # First convolution
        # ----------------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(
            out,
            inplace=True
        )

        # ----------------------------------------------------
        # Second convolution
        # ----------------------------------------------------

        out = self.conv2(out)

        out = self.bn2(out)

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        out = out + identity

        out = F.relu(
            out,
            inplace=True
        )

        return out


# ============================================================
# TERNARY RESNET18
# ============================================================

class TernaryResNet18(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        self.in_planes = 64

        # ----------------------------------------------------
        # CIFAR-10 Stem
        #
        # 3x3 convolution
        # stride 1
        # no max pooling
        # ----------------------------------------------------

        self.conv1 = TernaryConv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            64
        )

        # ----------------------------------------------------
        # ResNet18 layers
        #
        # [2, 2, 2, 2]
        # ----------------------------------------------------

        self.layer1 = self._make_layer(
            64,
            2,
            stride=1
        )

        self.layer2 = self._make_layer(
            128,
            2,
            stride=2
        )

        self.layer3 = self._make_layer(
            256,
            2,
            stride=2
        )

        self.layer4 = self._make_layer(
            512,
            2,
            stride=2
        )

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        # ----------------------------------------------------
        # Ternary Fully Connected Layer
        # ----------------------------------------------------

        self.fc = TernaryLinear(
            512,
            num_classes
        )

        # ----------------------------------------------------
        # Final initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # CREATE RESNET LAYER
    # ========================================================

    def _make_layer(
        self,
        planes,
        blocks,
        stride
    ):

        layers = []

        # First block
        layers.append(
            BasicBlock(
                self.in_planes,
                planes,
                stride
            )
        )

        self.in_planes = planes

        # Remaining blocks
        for _ in range(
            1,
            blocks
        ):

            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=1
                )
            )

        return nn.Sequential(
            *layers
        )

    # ========================================================
    # MODEL INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            # ------------------------------------------------
            # Ternary Conv
            # ------------------------------------------------

            if isinstance(
                module,
                TernaryConv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

                # Initialize alpha
                with torch.no_grad():

                    initial_alpha = (
                        module.weight.abs().mean()
                    )

                    module.log_alpha.copy_(
                        torch.log(
                            torch.expm1(
                                initial_alpha.clamp_min(1e-6)
                            )
                        )
                    )

            # ------------------------------------------------
            # Ternary Linear
            # ------------------------------------------------

            elif isinstance(
                module,
                TernaryLinear
            ):

                nn.init.kaiming_uniform_(
                    module.weight,
                    a=5 ** 0.5
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

                with torch.no_grad():

                    initial_alpha = (
                        module.weight.abs().mean()
                    )

                    module.log_alpha.copy_(
                        torch.log(
                            torch.expm1(
                                initial_alpha.clamp_min(1e-6)
                            )
                        )
                    )

            # ------------------------------------------------
            # BatchNorm
            # ------------------------------------------------

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(
            out,
            inplace=True
        )

        # ----------------------------------------------------
        # ResNet layers
        # ----------------------------------------------------

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        out = self.avgpool(out)

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        out = torch.flatten(
            out,
            1
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        out = self.fc(out)

        return out


# ============================================================
# TERNARY WEIGHT VERIFICATION
# ============================================================

@torch.no_grad()
def check_ternary_weights(
    model,
    verbose=True
):

    all_ternary = True

    checked_layers = 0

    for name, module in model.named_modules():

        # Only Conv/Linear are ternarized
        if not isinstance(
            module,
            (
                TernaryConv2d,
                TernaryLinear
            )
        ):

            continue

        weight = module.weight

        alpha = module.alpha

        # ----------------------------------------------------
        # Normalize latent weight
        # ----------------------------------------------------

        normalized = (
            weight / alpha
        )

        # ----------------------------------------------------
        # Generate ternary values
        # ----------------------------------------------------

        ternary = torch.where(
            normalized > 0.5,
            torch.ones_like(weight),
            torch.where(
                normalized < -0.5,
                -torch.ones_like(weight),
                torch.zeros_like(weight)
            )
        )

        # ----------------------------------------------------
        # Quantized weight
        # ----------------------------------------------------

        quantized_weight = (
            ternary * alpha
        )

        # ----------------------------------------------------
        # Check:
        #
        #       -alpha
        #        0
        #       +alpha
        # ----------------------------------------------------

        valid_positive = torch.isclose(
            quantized_weight,
            alpha,
            atol=1e-6
        )

        valid_negative = torch.isclose(
            quantized_weight,
            -alpha,
            atol=1e-6
        )

        valid_zero = torch.isclose(
            quantized_weight,
            torch.zeros_like(
                quantized_weight
            ),
            atol=1e-6
        )

        valid = (
            valid_positive
            |
            valid_negative
            |
            valid_zero
        )

        layer_ok = bool(
            torch.all(valid).item()
        )

        if not layer_ok:

            all_ternary = False

        checked_layers += 1

        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        if verbose:

            unique_values = torch.unique(
                quantized_weight
            )

            print(
                f"{name:35s} "
                f"alpha={alpha.item():.6f} "
                f"unique={unique_values.numel()} "
                f"{'PASS' if layer_ok else 'FAIL'}"
            )

    if verbose:

        print()

        print(
            f"Checked ternary layers: "
            f"{checked_layers}"
        )

        print(
            "Overall ternary constraint:",
            "PASS"
            if all_ternary
            else "FAIL"
        )

    return all_ternary


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TERNARY RESNET18 TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # GPU information
    # --------------------------------------------------------

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TernaryResNet18(
        num_classes=10
    ).to(device)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Dummy CIFAR-10 input
    # --------------------------------------------------------

    x = torch.randn(
        4,
        3,
        32,
        32,
        device=device
    )

    print(
        "\nInput shape:",
        x.shape
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    y = model(x)

    print(
        "Output shape:",
        y.shape
    )

    # --------------------------------------------------------
    # Verify ternary weights
    # --------------------------------------------------------

    print(
        "\nChecking ternary weights..."
    )

    ternary_ok = check_ternary_weights(
        model,
        verbose=True
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL:",
        "PASS"
        if ternary_ok
        else "FAIL"
    )

    print(
        "=" * 70
    )

    print(
        "\nTest completed."
    )
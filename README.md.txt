# Pytorch Installation on Raspberry Pi 4

Environment Configuration:

- [Raspbian GNU/Linux 10 (buster)](https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-05-28/2021-05-07-raspios-buster-armhf-lite.zip)
-- You can verify by 
    ```sh
    cat /etc/os-release
    ```
    -- If not installed, I recommend downloading the [lite version](https://downloads.raspberrypi.org/raspios_lite_armhf/images/raspios_lite_armhf-2021-05-28/2021-05-07-raspios-buster-armhf-lite.zip), and use the [Raspberry Pi Imager](https://downloads.raspberrypi.org/imager/imager_latest.dmg) to burn the image on the SD Card by selecting the 'Choose OS' --> 'Use Custom'

- ARMv7 Processor rev 3 (v7l)
    ```sh
    cat /proc/cpuinfo
    ```

## Installing Torch from [ljk53](https://github.com/ljk53/pytorch-rpi/blob/master/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl)
- Update & Upgrade
    ```sh
    sudo apt-get update
    sudo apt-get upgrade
    ```
- Install Dependencies
    ```sh
    sudo apt-get install ninja-build git cmake
    sudo apt-get install libopenmpi-dev libomp-dev ccache
    sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev libopenblas-base libatlas-base-dev
    sudo -H pip3 install -U --user wheel mock pillow
    ```
- Upgrade setuptools
    ```sh
    sudo -H pip3 install -U setuptools
    ```
- Download the pytorch wheel
    ```sh
    mkdir <DIRECTORY_OF_CHOICE>
    cd <DIR_PATH>
    wget https://github.com/ljk53/pytorch-rpi/blob/master/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl
    ```
- Install wheel
    ```sh
    pip3 install torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl 
    ```
- Quick test to make sure it works
    ```sh
    python3
    >>> import torch
    >>> print(torch.randn(1, 1, 32, 32))
    ```

## Install torchvision from [nmilosev](https://github.com/nmilosev/pytorch-arm-builds/blob/master/torchvision-0.4.0a0%2Bd31eafa-cp37-cp37m-linux_armv7l.whl)
- Download & Install wheel
    ```sh
    cd <DIR_PATH>
    wget https://github.com/nmilosev/pytorch-arm-builds/blob/master/torchvision-0.4.0a0%2Bd31eafa-cp37-cp37m-linux_armv7l.whl
    pip3 install torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl 
    ```
- Quick test
    ```sh
    python3
    >>> import torchvision
    ```
- If it fails, you may install additional libs
    ```sh
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
    ```
    
## Running Flask
- Install requirements.txt (this file contains libs for both camera capture and flask)
    ```sh
    alembic==1.4.1
    boto3==1.17.110
    botocore==1.20.110
    certifi==2021.5.30
    charset-normalizer==2.0.4
    click==8.0.1
    cloudpickle==1.6.0
    databricks-cli==0.15.0
    dataclasses==0.6
    docker==5.0.0
    entrypoints==0.3
    Flask==2.0.1
    future==0.18.2
    gitdb==4.0.7
    GitPython==3.1.18
    gunicorn==20.1.0
    idna==3.2
    importlib-metadata==4.6.4
    itsdangerous==2.0.1
    Jinja2==3.0.1
    jmespath==0.10.0
    Mako==1.1.5
    MarkupSafe==2.0.1
    mlflow==1.20.1
    numpy==1.21.2
    packaging==21.0
    pandas==1.1.5
    picamera==1.13
    Pillow==8.3.1
    prometheus-client==0.11.0
    prometheus-flask-exporter==0.18.2
    protobuf==3.17.3
    pyparsing==2.4.7
    python-dateutil==2.8.1
    python-editor==1.0.4
    pytz==2021.1
    PyYAML==5.4.1
    querystring-parser==1.2.4
    requests==2.26.0
    s3transfer==0.4.2
    six==1.16.0
    smmap==4.0.0
    SQLAlchemy==1.4.23
    sqlparse==0.4.1
    tabulate==0.8.9
    typing-extensions==3.10.0.0
    urllib3==1.26.6
    websocket-client==1.2.1
    Werkzeug==2.0.1
    zipp==3.5.0
    ```
    ```sh
    pip3 install -r requirements.txt
    ```
- Copying the flask_server_api.py file
    ```sh
    import base64
    import io
    import tarfile
    
    import requests
    from flask import Flask, request, jsonify
    import torch
    import mlflow
    import numpy
    from torchvision import datasets, models, transforms
    from PIL import Image, ImageDraw
    from picamera import PiCamera
    from time import sleep
    import datetime as dt
    import sys
    import subprocess
    import os
    import json
    from torch.hub import load_state_dict_from_url
    from torchvision.models import mobilenet_v2 as mobilenetv2
    
    app = Flask(__name__)
    @app.route('/', methods=['GET'])
    def alive():
        timestamp = dt.datetime.now()
        return jsonify({"response":"API is alive", "status":"200", "url":"frostydew1905.cotunnel.com", "timestamp": timestamp})
    
    @app.route('/api/send_url/', methods=['GET', 'POST'])
    def get_url():
        content = request.json
        model = content["model"]
        version = content["version"]
        url = content["url"]
        status = "SUCCESS"
        response = requests.get(url, stream=True)
        target_path = '/home/pi/compressed/model.tar.gz'
        timestamp = dt.datetime.now()
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        return jsonify({"url": url, "model": model, "version": version, "status": status, "timestamp": timestamp})
    
    @app.route('/api/artifacts/', methods=['GET', 'POST'])
    def download_artifacts():
        content = request.json
        model = content["model"]
        version = content["version"]
        data = content["data"]
        string_to_b64 = data.encode('utf-8')
        b64_to_bytes = base64.b64decode(string_to_b64)
        file_like_object = io.BytesIO(b64_to_bytes)
        tar = tarfile.open(fileobj=file_like_object, mode='r:gz')
        dest_path = '/home/pi/artifacts/' + model + '_' + version + '/'
        tar.extractall(path=dest_path)
        timestamp = dt.datetime.now()
        return jsonify({"model": model, "version": version, "status": "production", "dest_path": dest_path, "timestamp": timestamp})
    
    @app.route('/api/inference/', methods=['GET'])
    def infer():
        model_path = '/home/pi/artifacts/MobileNetV3_100/MobileNetV3/data/model.pth'
        labels = json.dumps({"0": "empty", "1": "box"})
        CURRENT_DATE = dt.datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
        camera = PiCamera()
        camera.resolution = (600, 600)
        camera.framerate = 15
        camera.start_preview()
        sleep(10)
    
        image_filepath = '/home/pi/images/' + CURRENT_DATE + '.jpg'
        camera.capture(image_filepath)
        camera.stop_preview()
        camera.close()
        test_image = image_filepath
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        labels = json.loads(labels)
        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(test_image)
        image = data_transform(img).unsqueeze(0).to(device)
        model = torch.load(model_path, pickle_module=mlflow.pytorch.pickle_module, map_location=torch.device('cpu'))
        model.to(device)
        model.eval()
        out = model(image)
        # print(out)
        print(out.argmax().item())
        print("Predicted class is: {}".format(labels[str(out.argmax().item())]))
        prediction = labels[str(out.argmax().item())]
        timestamp = dt.datetime.now()
        d1 = ImageDraw.Draw(img)
        text = "Prediction: " + prediction
        d1.text((28, 36), text, fill=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, 'jpeg')
        buf.seek(0)
        img_bytes = buf.read()
        buf.close()
        return jsonify({"predicted": prediction, "timestamp": timestamp, "image": base64.b64encode(img_bytes).decode('utf-8')})
    
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port='8080', debug=True)
    ```
- Testing with Postman
    -- Send a GET request to the raspberry pi local address (make sure to disconnect from VPN). You can locate it by:
        ```sh
        arp-scan --localhost 
        ```
        If successful, you should see the following after sending a GET request using <RPI IP>/:
        ```sh
        {
        "response": "API is alive",
        "status": "200",
        "timestamp": "Mon, 04 Oct 2021 22:43:10 GMT",
        "url": "frostydew1905.cotunnel.com"
        }
        ```
- Getting /api/inference endpoint to work
    -- Create two folders
    ```sh
    mkdir /home/pi/artifacts
    mkdir /home/pi/images
    ```
    -- Verify if the pre-trained model is in the modules (it won't likely be)
    ```sh
    python3
    >>> import torchvision
    >>> torchvision.__file__
    ```
    -- Copy the filepath and replace __init__.py by models, so your path should look like:
    ```sh
    cd /home/pi/.local/lib/python3.7/site-packages/torchvision/models/
    ```
    If you don't mobilenetv2 or mobilenetv3 we need to manually copy it over. You can use nano mobilenetv2.py and mobilenetv3.py to copy the following:
    -- mobilenetv2.py:
    ```sh
    import torch
    from torch import nn
    from torch import Tensor
    from .utils import load_state_dict_from_url
    from typing import Callable, Any, Optional, List
    
    
    __all__ = ['MobileNetV2', 'mobilenet_v2']
    
    
    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    }
    
    
    def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    
    class ConvBNActivation(nn.Sequential):
        def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
        ) -> None:
            padding = (kernel_size - 1) // 2 * dilation
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if activation_layer is None:
                activation_layer = nn.ReLU6
            super().__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                          bias=False),
                norm_layer(out_planes),
                activation_layer(inplace=True)
            )
            self.out_channels = out_planes
    
    
    # necessary for backwards compatibility
    ConvBNReLU = ConvBNActivation
    
    
    class InvertedResidual(nn.Module):
        def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            super(InvertedResidual, self).__init__()
            self.stride = stride
            assert stride in [1, 2]
    
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
    
            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup
    
            layers: List[nn.Module] = []
            if expand_ratio != 1:
                # pw
                layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ])
            self.conv = nn.Sequential(*layers)
            self.out_channels = oup
            self._is_cn = stride > 1
    
        def forward(self, x: Tensor) -> Tensor:
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)
    
    
    class MobileNetV2(nn.Module):
        def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
        ) -> None:
            """
            MobileNet V2 main class
    
            Args:
                num_classes (int): Number of classes
                width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
                inverted_residual_setting: Network structure
                round_nearest (int): Round the number of channels in each layer to be a multiple of this number
                Set to 1 to turn off rounding
                block: Module specifying inverted residual building block for mobilenet
                norm_layer: Module specifying the normalization layer to use
    
            """
            super(MobileNetV2, self).__init__()
    
            if block is None:
                block = InvertedResidual
    
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
    
            input_channel = 32
            last_channel = 1280
    
            if inverted_residual_setting is None:
                inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1],
                ]
    
            # only check the first element, assuming user knows t,c,n,s are required
            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
                raise ValueError("inverted_residual_setting should be non-empty "
                                 "or a 4-element list, got {}".format(inverted_residual_setting))
    
            # building first layer
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                    input_channel = output_channel
            # building last several layers
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
            # make it nn.Sequential
            self.features = nn.Sequential(*features)
    
            # building classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )
    
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
    
        def _forward_impl(self, x: Tensor) -> Tensor:
            # This exists since TorchScript doesn't support inheritance, so the superclass method
            # (this one) needs to have a name other than `forward` that can be accessed in a subclass
            x = self.features(x)
            # Cannot use "squeeze" as batch-size can be 1
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)
    
    
    def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
        """
        Constructs a MobileNetV2 architecture from
        `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        model = MobileNetV2(**kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        return model
    ```
    -- mobilenetv3.py:
    ```sh
    import torch

    from functools import partial
    from torch import nn, Tensor
    from torch.nn import functional as F
    from typing import Any, Callable, Dict, List, Optional, Sequence
    
    from torchvision.models.utils import load_state_dict_from_url
    from torchvision.models.mobilenetv2 import _make_divisible, ConvBNActivation
    
    
    __all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]
    
    
    model_urls = {
        "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
    }
    
    
    class SqueezeExcitation(nn.Module):
        # Implemented as described at Figure 4 of the MobileNetV3 paper
        def __init__(self, input_channels: int, squeeze_factor: int = 4):
            super().__init__()
            squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
            self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
    
        def _scale(self, input: Tensor, inplace: bool) -> Tensor:
            scale = F.adaptive_avg_pool2d(input, 1)
            scale = self.fc1(scale)
            scale = self.relu(scale)
            scale = self.fc2(scale)
            return F.hardsigmoid(scale, inplace=inplace)
    
        def forward(self, input: Tensor) -> Tensor:
            scale = self._scale(input, True)
            return scale * input
    
    
    class InvertedResidualConfig:
        # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
        def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                     activation: str, stride: int, dilation: int, width_mult: float):
            self.input_channels = self.adjust_channels(input_channels, width_mult)
            self.kernel = kernel
            self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
            self.out_channels = self.adjust_channels(out_channels, width_mult)
            self.use_se = use_se
            self.use_hs = activation == "HS"
            self.stride = stride
            self.dilation = dilation
    
        @staticmethod
        def adjust_channels(channels: int, width_mult: float):
            return _make_divisible(channels * width_mult, 8)
    
    
    class InvertedResidual(nn.Module):
        # Implemented as described at section 5 of MobileNetV3 paper
        def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                     se_layer: Callable[..., nn.Module] = SqueezeExcitation):
            super().__init__()
            if not (1 <= cnf.stride <= 2):
                raise ValueError('illegal stride value')
    
            self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
    
            layers: List[nn.Module] = []
            activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
    
            # expand
            if cnf.expanded_channels != cnf.input_channels:
                layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                               norm_layer=norm_layer, activation_layer=activation_layer))
    
            # depthwise
            stride = 1 if cnf.dilation > 1 else cnf.stride
            layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                           stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                           norm_layer=norm_layer, activation_layer=activation_layer))
            if cnf.use_se:
                layers.append(se_layer(cnf.expanded_channels))
    
            # project
            layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.Identity))
    
            self.block = nn.Sequential(*layers)
            self.out_channels = cnf.out_channels
            self._is_cn = cnf.stride > 1
    
        def forward(self, input: Tensor) -> Tensor:
            result = self.block(input)
            if self.use_res_connect:
                result += input
            return result
    
    
    class MobileNetV3(nn.Module):
    
        def __init__(
                self,
                inverted_residual_setting: List[InvertedResidualConfig],
                last_channel: int,
                num_classes: int = 1000,
                block: Optional[Callable[..., nn.Module]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                **kwargs: Any
        ) -> None:
            """
            MobileNet V3 main class
    
            Args:
                inverted_residual_setting (List[InvertedResidualConfig]): Network structure
                last_channel (int): The number of channels on the penultimate layer
                num_classes (int): Number of classes
                block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
                norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            """
            super().__init__()
    
            if not inverted_residual_setting:
                raise ValueError("The inverted_residual_setting should not be empty")
            elif not (isinstance(inverted_residual_setting, Sequence) and
                      all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
                raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")
    
            if block is None:
                block = InvertedResidual
    
            if norm_layer is None:
                norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    
            layers: List[nn.Module] = []
    
            # building first layer
            firstconv_output_channels = inverted_residual_setting[0].input_channels
            layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish))
    
            # building inverted residual blocks
            for cnf in inverted_residual_setting:
                layers.append(block(cnf, norm_layer))
    
            # building last several layers
            lastconv_input_channels = inverted_residual_setting[-1].out_channels
            lastconv_output_channels = 6 * lastconv_input_channels
            layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=nn.Hardswish))
    
            self.features = nn.Sequential(*layers)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(lastconv_output_channels, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(last_channel, num_classes),
            )
    
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
    
        def _forward_impl(self, x: Tensor) -> Tensor:
            x = self.features(x)
    
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
    
            x = self.classifier(x)
    
            return x
    
        def forward(self, x: Tensor) -> Tensor:
            return self._forward_impl(x)
    
    
    def _mobilenet_v3_conf(arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
                           **kwargs: Any):
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1
    
        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)
    
        if arch == "mobilenet_v3_large":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1280 // reduce_divider)  # C5
        elif arch == "mobilenet_v3_small":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1024 // reduce_divider)  # C5
        else:
            raise ValueError("Unsupported model type {}".format(arch))
    
        return inverted_residual_setting, last_channel
    
    
    def _mobilenet_v3_model(
        arch: str,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        pretrained: bool,
        progress: bool,
        **kwargs: Any
    ):
        model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
        if pretrained:
            if model_urls.get(arch, None) is None:
                raise ValueError("No checkpoint is available for model type {}".format(arch))
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        return model
    
    
    
    def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
        """
        Constructs a large MobileNetV3 architecture from
        `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        arch = "mobilenet_v3_large"
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
        return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
    
    
    
    
    def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
        """
        Constructs a small MobileNetV3 architecture from
        `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.
    
        Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
        """
        arch = "mobilenet_v3_small"
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
        return _mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)

    ```
    -- Let's [download an example of an artifact](https://drive.google.com/drive/folders/1DuSfiI60ORgVqJtft2Jv6lF9qZgUL7r0?usp=sharing) and transfer from local to the Raspberrypi
    ```sh
    scp -r <LOCAL_DIR>/MobileNetV3_100 pi@raspberrypi.local:/home/pi/artifacts/
    ```
    -- Finally, we can run our flask server by:
    ```sh
    python3 flask_server.py
    ```
    -- And make a GET call from Postman on the inference endpoint:
        ```
        <RPI_IP>/api/inference/
        ```
    If successful, there response will look like:
    ```sh
        {
        "image": "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70d6BhRRR1oAKKSigBaSijNAgooNFABQaKKACiiigBe1JRRQAUUlLQA2nCmdKctICTNOpgpwpjHCnCminCmA4UopopwpgOFOFMFOApjHA08UwU4UwHinCmDrThTAcKfTBThTAcKeDxTM08GmA4Ypw60wc04UxjxT160wU9eKYDqdTRS0wJB2p49qjWpAcjFMY9Tkc0UgPGKKYHJ0UUVxEiUUtJTAKKKSgBaKTNFAgoooNAAaSiigAoopKACiiigA7UUGigApKKKACiikoAKKKSgQvakoooAKKO9BpAFFFFABRSUUxhRQKKBBRRRQAUGg0UAFFFJQAdaWkzRQAtJRRQAUUUUAFFFFADT1pwph4NOBpAPFOFMFOFMB4pwNNpwpjHClFNFLTAeKdTBThTGPBpwpgpwpgPFOFMHWnCmA8GnA0wGnCmA4c09aaKcKYD+lKOtNp4xTGLTwaYDzTxTAeOaU0lOABxTAcORT14zTB1pwpjJF9aKRaKAOV602lorkIEooooAKDRSUAFHSiigApDS0lABR1oooAKSiigAooooASiiigApKKKACiikNAB2oopM8UALSE0ZpaBBRSUfWkAdKKKKACiiimMKSiigQUtJRQAppKM8UUAFFFFABRRiigAopKWgAooooASlpKKAGt1pwprdaUGgB4pwpgpwoAeKdTRSg0xjxSim0opgPHNOpgp1ADxmnCmCnCmMcDTqaBThVAP7U4dKYKcKYD85p1NBpw5pgPFOFMHWnDIpjHinimYp2KYDwaeOlMUZp464pgOFOFN6U4HNMY9egooGeKKYHJ0UdqK4yApKXFJQAUUUUAFFFJQAUUUUAHSkooNABRRSUAFFFFAAaSijNABSdaWkoEFFFJQAtBpKKACilpKACiikoAWjvSd6WgBKKKO1AB2ooooAKKOKKACiig0AFGaKKAEpaSigApaSloAKKSloASiiigBrUChulIKAHg08UwU4UAPFOFMBp1MB4pRTRSimMePenCmCnUAOFPFMFOBpgPBp2aYOtOFMY8U4etMFPFUA4cmnDrTB1qQUwHLxTx2pnFOpgPp3pTRSjrTGSA04UwHtTlHNMCSlHIpg61IvApjHiigdKKYHJ0CiiuMgKSlpO9ABRRRQAlFFFABRRRQAlFFHagApKKSgBaSlpKACikooEFFFGKAEozRR2oAKKKQ0AKaO1FFABSUtJQAUtJRQAUUUdKAEopaSgBaKKKAEpTSUtACdKXmkooAO9FFGaACiiigBaQUYooAKKKKAGt0pBTm+7TBQA8GnioxTxQA8U4U0UooAcKd0pgpwpgOFOHWmg0opjHjmnimClBpgPFOpgpwNMY9aeKaBTgcUwHCnA00U4VQDxTqaKcOlMB45Ap3SmgU4dKYx4HSnU3tSrTAepzUmajAwad3pjJAaKTrRTA5WijtRXGSJRS4pO9AgpKXFFACUUUUAFFFIaACiikoAKSlooASg0UUCCkpaKAEooooAKKKKAEooxRQAUYoooAKSlxRQAmKKWigBKKWigBKKWigBO1FFAFABRRiigAooxRQAlFL1oxQAlLSUUAFFFGKACiiigBD0pgp5qOgB608UwU4UAPpaaKcKAHClFIKWmA4U4cUynCmA4dKeKaKcKYxwNOzTRTh1pgPBNOHNMFSCmhi08HmmA5pw61QDxThUY61IKYEgIpw60xacOtMY8HNPXpUfU1JTAcDThTR2p3XFMY8cHNFGccUUwOVoxS0VyEjaKWigQnWg0tJQAY5pKWigBMUdaKDQAlFFFIQlGaKKACkNKaSgAooooASloozQAlFLSUAJRilooASilooAKSlo/CgBKKWigBMUGloxQAlFLRimAlFLRQAlApaSgApOadRQA00Up5pMc0AFJS0UAJS0UlABSUtFACGo+9S1GeppAOFOFMFPFADhTu9MFOBoAdThTaWmA4HmnDrTBTutMB4pwNMFOFMY4Gng0wU4UwJBTgaYpp+e9MY8U7GKYDTwaYCjrTwaYKf2pgPWpB1pi04daoB1PpAKWmMcOtSDgVGtSA8UwHAd6KF6UUxnLUUUVyECUUtFACUlLRQAlFFFACUUUUAJRRRQAUlLRQISilxSUAFFFFACUUtGKADFJS0UAJRS0YoASil7UUAJRS0YoATFFLiigBKMUtFACYoxS4ooASilooATFJilxRQAnSjFLijFACYpMU6koASilxRigBuKKXFFACUYpTSUAJUbfeqXmo2HNIAFPFMHWnigBwpaaKcKYDhS0lLQA4UopopwpgOpwpopwpjHCnimU4cUwHDrTxTFp460wH0oPNNGc08dKYxR1qQVGOtPHWmBItPFMBwKcDVAPzxT1qMdKetMZIKcOlMFOpgSKeKKanXmimM5mkxS0lchAUUUUAJRRRQAhopaSgApKWkoASjrSmigBKKKKBBRRRQAUUUUAJiloooASloooAKKKWgBtFOooATFGKXGKMUwEAopcUUAJiilooATFGKWjFACUUuKSgBKXFLRQA2ilooATFBpaSgBKMUtIaACkpaKQCYpDS0YFACVG45zUlMfrQA0ZpwptOpAOFOFNp2aAFFLikFLTAcKWkFKOKYDhThTRThTGOFOpopwNMBwNPFMFOFMCRelOFRqakHNMYoJp69aaKcOtMCQU/wBqZTx1pgOHBp1NH3qeMGqGPWnjBpi08EdaYDgMcUUKc80UDOYoopK5SAoNFFACUUtIaAEopaSgApKWigBKKMUUAFFFFAhKKWigApKWlpgJijBpcUUAJRS4ooAKTmlxRQAUYopaAEopcUYoASilxRigBMUUuKXFADcUU6koASjFLRQAmKKWigBuKWlpKAEpMU7FFADaKWigBvainUmKAG4oxS0UANpknSpKZJ92kBGKeKaKWkA4UopKcKYCilpBSigBRTqQUopgOFOFNFOFMY4U4CmjrThTAeKd2pgNOFMBy09aatOFMY8cCnio6etMB4PNPUZNMWnrTAeKeKjzzTx7VQx604DmminjkUwHr1ooUjvRQM5iiiiuYgTpRQaKACg0UlABSUtJSAKKKKYhKKWigBKKWigBKXFFFABRilooASloxS0AJRS4oxTGJilxS4oxQAmKMUuOKWgBuKKdRQAmKMUuKXFADcUYp2KKAG4oxTqSgBMUYp2KSgBCKTFOxSUCExSYp2KCKAG0Yp1JigBtGKdikoAbQacRSYoAbRSmkpAIetMfpUlNccGgCGnCm05aQCgU4UgpaAFpRSUooAdS0g4paYCinU0U+mMcKdTBTqYDwKcKbSimBIp4pwqOpBTGOHIp4pi9aeOtMB4NSCox6U8UwHCnrTAc09elMY4c08cUwdadjNUBJ3opoFFAHN0UvakxXMSJS0YooASjvQaKAEopTSYoAKKKKAEpe1FGKAEpcUYpaAEoxS0UAIKWjFKBTAKMUuKKAEoxS4paAEpRRS0AJRSgUUwCilxRigBMUYp1GKAG4oxS4paAG4oxTsUYoAbiilxRigBuKMU7FGKAG4oxS0d+lADcUmKdRigBuKKXFGKQhppKdikxQA3FJinYpKAEIprDin009DSAgxSik7mlFIBRThSCnCgA70ooApaAF60vpSClFMB1KKQUtMY4U8UwU8daYDgKUUmacKYDgM08cUwU4GmA8U4dcmmDinjpTGPB704c00dKcOtMCT0p4NR5p4PamMeKePSmACnjimA5eKKBzRTA5uiiiuYkSilpKACkpcUUAJSUtFACUUtFABRRRQAUUdKWgAo7UtFMAoxRS4oASlxS0d6YBijFLRigBKWlxRigAxRilxRigBKXHFLiimAmOKMU7FFAxMUmKdRigQ2lpcUYoAbQRS4ooAaaKdiigBuKTFOxSYoATFJTqSkA2jFLijFADaTFOpCKBDaTFONJQAlIaUigjikBXI5oFK3U0CpAWlHSkpe1AC0opBS96YC0ooooAcKcOaZThTGOFOHWmg04UwHinCmU7NMB/anCminUwHU+mCn5pjHinCminL1pgPHrThSdqcKYxymnimDpTx2pgPGMUUmeKKYznKKWkrmICkpaKAEooooAKKKKACijvRigAoxS0d6YCUtFLQAUUUtACUoopaYBRS4ooAKWiigAxS0YpcZpgJS4pcUYoGJS4paKAEoxS4paYDcUYp1JQAmKMUuKMUAJijFLRikIbQRS4ooAbRS0UANNJTqKAG4zSU6kxQA3FIRTiKTFIBtJin4ptACGkPSnYpMUhFdvvUgpz8MaaKQCiloFKKQCilFJSjimAUtJTgaAFFOHWminDrTGO6UopopwpgOHWnCminimA4cU4U3vTxTAXFPHApop60xju1PAFR0/PFMB9PHamDpTxTGOHWn9KjHWpB0pgOzRQBxRTGc6aKKSuYgWkoFFABSUtFACUUtFACUtJS0wCiiloASnYpKWgAAooxSimAdqXFJS0AGKWiloAKMUUopgFKBxRSigAopaMUxhRiloxQAUYopRQAlJinYooASjFLRQA3FFLRQAlJTqSgQ3FFLiigBppMU6kxQAlJS4opAJTadSGgBMU2nYoIpANpDS4oOaBFeQfNTRT5B81NqWAd6Wkp1ABSikpaAFpQKSlFMBw6Uo60gpRQAtOpvSnDpTGOHSniminL0pgOFPFRipFpgLnNPHFM704Uxj15p/amKKkApgOHSnU1etO70xjwKcKSnUwHAccUUDpRTGc7RS0lc5AmKWiigBKWiigBKKKWgBKWiigAooooAWigUopgFKKMUYoGFLRRQIWlooFMYUoFFLQAUoopaYCYpaWigA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",
        "predicted": "empty",
        "timestamp": "Mon, 04 Oct 2021 23:25:05 GMT"
    }
    ```   
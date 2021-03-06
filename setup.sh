#!/bin/sh
cpu_v=$(cat "/proc/cpuinfo")
version=${cpu_v:31:1}
case $version in
	7)
		echo "Sys: 64 bits"
		bits="64"
		;;
	*)
		echo "Sys: 32 bits"
		bits="32"
		;;
esac


echo "----------------------------"
echo "Installing Pytorch"
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install wget
sudo apt-get install python3-pip
echo "Installing Dependencies"
sudo apt-get install ninja-build git cmake
sudo apt-get install libopenmpi-dev libomp-dev ccache
sudo apt-get install libopenblas-dev libblas-dev libeigen3-dev libopenblas-base libatlas-base-dev
sudo -H pip3 install -U --user wheel mock pillow
sudo -H pip3 install -U setuptools
dir="$(pwd)/download"
echo "All Files will be downloaded to: $dir"
if [[ ! -e $dir ]]; then
    mkdir $dir
elif [[ ! -d $dir ]]; then
    echo "$dir already exists but is not a directory" 1>&2
fi

TORCHFILE="$dir/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl"
if [ -f "$TORCHFILE" ]; then
    echo "$TORCHFILE already exists."
else 
    echo "$TORCHFILE does not exist."
    wget https://github.com/ljk53/pytorch-rpi/raw/master/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl -P $dir
fi
pip3 install "$dir/torch-1.7.0a0-cp37-cp37m-linux_armv7l.whl"

test_dir="$(pwd)/test"
if [[ ! -e $test_dir ]]; then
    mkdir $test_dir
elif [[ ! -d $test_dir ]]; then
    echo "$test_dir already exists but is not a directory" 1>&2
fi

PYFILE="$test_dir/pytorch_test.py"
if [ -f "$PYFILE" ]; then
    echo "$PYFILE already exists."
else 
    echo "$PYFILE does not exist."
    echo "import torch" >> "$test_dir/pytorch_test.py"
    echo "print(torch.randn(1, 1, 32, 32))" >> "$test_dir/pytorch_test.py"
fi

echo "----------------------------"
echo "Testing Torch"
python3 "$test_dir/pytorch_test.py"
echo "----------------------------"
echo "Installing Torchvision"
VISIONFILE="$dir/torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl"
if [ -f "$VISIONFILE" ]; then
    echo "$VISIONFILE already exists."
else 
    echo "$VISIONFILE does not exist."
    wget https://github.com/nmilosev/pytorch-arm-builds/raw/master/torchvision-0.4.0a0%2Bd31eafa-cp37-cp37m-linux_armv7l.whl -P $dir
fi
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
pip3 install "$dir/torchvision-0.4.0a0+d31eafa-cp37-cp37m-linux_armv7l.whl"

echo "----------------------------"
echo "Testing Torchvision"
VPYFILE="$test_dir/torchvision_test.py"
if [ -f "$VPYFILE" ]; then
    echo "$VPYFILE already exists."
else 
    echo "$VPYFILE does not exist."
    echo "import torchvision" >> "$test_dir/torchvision_test.py"
    echo "print('Torchvision imported successfully')" >> "$test_dir/torchvision_test.py"
fi

python3 "$test_dir/torchvision_test.py"
echo "----------------------------"
echo "Downloading source and artifacts"
source_dir="$(pwd)/source"
if [[ ! -e $source_dir ]]; then
    mkdir $source_dir
elif [[ ! -d $source_dir ]]; then
    echo "$source_dir already exists but is not a directory" 1>&2
fi

RFILE="$source_dir/requirements.txt"
if [[ -f "$RFILE" ]]; then
    echo "$RFILE already exists."
    echo "Removing $RFILE"
    rm $RFILE
    echo "Downloading new file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/requirements.txt -P $source_dir
else 
    echo "$RFILE does not exist."
    echo "Downloading new file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/requirements.txt -P $source_dir
fi

WFILE="$source_dir/webserver.py"
if [[ -f "$WFILE" ]]; then
    echo "$WFILE already exists."
    echo "Removing $WFILE"
    rm $WFILE
    echo "Downloading new file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/webserver.py -P $source_dir
else 
    echo "$WFILE does not exist."
    echo "Downloading new file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/webserver.py -P $source_dir
fi

artifacts_dir="$(pwd)/artifacts"
if [[ ! -e $artifacts_dir ]]; then
    mkdir $artifacts_dir
elif [[ ! -d $artifacts_dir ]]; then
    echo "$artifacts_dir already exists but is not a directory" 1>&2
fi
pip3 install -r $source_dir/requirements.txt
wget https://github.com/pbdatabricks/edge_rpi/raw/main/MobileNetV3_100.zip -P $artifacts_dir
sudo apt install unzip
unzip "$artifacts_dir/MobileNetV3_100.zip" -d "$artifacts_dir"
rm "$artifacts_dir/MobileNetV3_100.zip"
echo "----------------------------"
echo "Install Dependencies"
img_dir="$(pwd)/images"
if [[ ! -e $img_dir ]]; then
    mkdir $img_dir
elif [[ ! -d $img_dir ]]; then
    echo "$img_dir already exists but is not a directory" 1>&2
fi
echo "import torchvision" >> "$test_dir/modules_test.py"
echo "print(torchvision.__file__)" >> "$test_dir/modules_test.py"
modules_path=$(python3 "$test_dir/modules_test.py")
replace="models/"
path=${modules_path//__init__.py/$replace}

MTFILE="$path/mobilenetv2.py"
if [[ -f "$MTFILE" ]]; then
    echo "$MTFILE already exists."
else 
    echo "$MTFILE does not exist."
    echo "Downloading $MTFILE file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/mobilenetv2.py -P $path
fi

MTHFILE="$path/mobilenetv3.py"
if [[ -f "$MTHFILE" ]]; then
    echo "$MTHFILE already exists."
else 
    echo "$MTHFILE does not exist."
    echo "Downloading $MTHFILE file"
    wget https://raw.githubusercontent.com/pbdatabricks/edge_rpi/main/mobilenetv3.py -P $path
fi
echo "----------------------------"
echo "Running Flask"
python3 $source_dir/webserver.py


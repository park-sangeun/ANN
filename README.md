# ANN
인공신경망 프로젝트_EMNIST를 이용한 CNN 최적화 및 분석
# 2024 ANN Tool Installation Guide

기준 스펙
CPU : AMD Ryzen 5950X
GPU : GeForce RTX GTX1660 Super
RAM : 128GB
SSD : 500G

### Anaconda 설치
1. Anaconda3 설치 전 다음 명령어를 실행
```
sudo apt-get update
sudo apt-get upgrade   
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
2. [Anaconda installer(Linux version)](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh) 다운로드한 후 다음 명령어를 실행 
```
sudo apt-get install python3 python python3-pip python-pip git
```
3. 압축 해제한 설치파일 실행
```
bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
```
-> ‘yes’, ‘enter’로 이용약관의 동의 및 설치 주소 확인 후, conda init 여부에는 ‘yes’

### NVIDIA DRIVER 설치
- nvidia-smi 명령 실행

### CUDA 설치 (CUDA 11.2.0)
(중요) python, Cuda, cuDNN, tensorflow 버전을 호환되도록 꼭 맞춰야 함
1. CUDA Toolkit 11.2.0 다운
   - [링크 접속](https://developer.nvidia.com/cuda-11.2.0-download-archive)
   - Linux -> x86_64 -> Ubuntu -> 18.04 -> runfile
```
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run
```
2. 설치 진행 시
* 어떤 경고 1 -> Continue
* 어떤 경고 2 -> Accept
* CUDA Installer se Agreement -> Driver에는 [X]를 해제한 후 Install (GPU driver는 따로 설치를 했거나, 추후 다시 한다)
* Please make sure that ~~ 가 뜨면 완료된 것

### CUDA 환경변수 설정
1. bashrc 접속
```
vi ~/.bashrc
```
2. i 눌러서 입력모드, 하단에 입력 후 :wq!
```
## ~/.bashrc
export PATH=$PATH:/usr/local/cuda-11.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
export CUDADIR=/usr/local/cuda-11.2
```
3. 설치 여부 확인
```
source ~/.bashrc
nvcc -V
```

### cuDNN 설치(CUDA 11.2 설치 완료 후 진행)
1. [링크 접속](https://developer.nvidia.com/rdp/cudnn-archive) 후 회원가입
2. cuDNN v8.1.0 for CUDA 11.2 -> cuDNN Library for Linux (x86_64) 다운로드
3. 아래 명령어 실행
```
# tgz 파일이 있는 곳으로 이동
cd /home/download 

# 압축 해제
tar xvzf cudnn-11.2-linux-x64-v8.1.0.77.tgz

# 복사
sudo cp cuda/include/cudnn* /usr/local/cuda-11.2/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64

# 권한 부여
sudo chmod a+r /usr/local/cuda-11.2/include/cudnn.h /usr/local/cuda-11.2/lib64/libcudnn*

# 링크 설정
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
sudo ln -sf /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8.1.0.77 /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8
```
4. 설치 확인
```
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```
5. 재부팅
```
sudo reboot
```

### 가상환경 구성하기(실습 위한 텐서플로우 설치)
1. 가상환경 구축
```
conda update -n base -c defaluts conda
conda env create -f environment.yml -n ANN
source ~/.bashrc
conda activate ANN
```
2. pip 설치
```
pip install --upgrade jupyter matplotlib numpy pandas pydot scipy scikit-learn opencv_python daal 
conda install python-graphviz
pip install --upgrade tensorflow_datasets tensorflow-hub tensorflow-addons keras-tuner scikeras
pip install tensorflow-gpu==2.8.0
pip install --upgrade grpcio
```

### 실습 .ipynb 파일 실행
1. 실습 파일 다운
2. 가상환경 활성화
```
conda activate ANN
```
3. python ~~.py or python ~~.ipynb 실행

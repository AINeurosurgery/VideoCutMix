


Commands

CUDA 11.7, Python=3.7, PyTorch=1.13.1

1. conda create --name=videocutmix python=3.7.16
2. conda activate videocutmix
3. pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
4. pip install PyYAML==5.4.1

Running

bash scripts/run.sh

Results will be available in results.json 


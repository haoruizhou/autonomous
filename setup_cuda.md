# Local CUDA Environment Setup

Follow these steps to set up the environment on your machine with the NVIDIA RTX 3080.

## Prerequisites
- **OS**: Linux (Ubuntu 20.04/22.04 recommended) or Windows (WSL2 recommended).
- **GPU**: NVIDIA RTX 3080.
- **Drivers**: Ensure NVIDIA drivers are installed (`nvidia-smi` works).
- **CUDA Toolkit**: 11.3 or 11.8 recommended.
- **Anaconda/Miniconda**.

## 1. Create Conda Environment
```bash
conda create -n droid_slam python=3.9 -y
conda activate droid_slam
```

## 2. Install PyTorch (CUDA Support)
Check your CUDA version (`nvcc --version`) and install the appropriate PyTorch. For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install Core Dependencies
```bash
pip install tqdm matplotlib opencv-python scipy argparse
```

## 4. Install Lietorch (Required for DROID)
```bash
git clone https://github.com/princeton-vl/lietorch.git
cd lietorch
python setup.py install
cd ..
```

## 5. Install DROID-SLAM
```bash
# You should already have this cloned via the project, if not:
# git clone https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM
python setup.py install
cd ..
```

## 6. Install Depth-Anything V2
```bash
# You should already have this clone
cd Depth-Anything-V2
pip install -r requirements.txt
cd ..
```

## 7. Run the Pipeline
```bash
python run_droid_cuda.py --video data/GX010035.MP4 --output results_cuda --vis
```

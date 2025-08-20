conda create -n talkingface python=3.10
conda activate talkingface

pip install -r requirements.txt
apt-get update && apt-get install -y ffmpeg

conda install -c conda-forge 'ffmpeg<7' -y
conda install torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
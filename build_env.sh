git clone https://github.com/dyyoungg/sglang-dev.git

git fetch

git checkout llavaomni

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 xformers==0.0.29.post2 --index-url https://download.pytorch.org/whl/cu126

cd ./sglang/python

CARGO_TARGET_DIR=/tmp/sglang_cargo_target pip install --no-build-isolation -e .
pip install sgl-kernel==0.3.17
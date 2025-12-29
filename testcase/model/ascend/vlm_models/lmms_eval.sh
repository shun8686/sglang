export PATH="/usr/local/Ascend/8.3.RC1/compiler/bishengir/bin:${PATH}"
pip uninstall lmms_eval
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install .
export PYTHONPATH=$(pwd)/lmms-eval:$PYTHONPATH
pip install pyproject.toml
pip install accelerate loguru sacrebleu evaluate sqlitedict tenacity pytablewriter dotenv torchaudio==2.8.0

hf download lmms-lab/MMMU --repo-type dataset


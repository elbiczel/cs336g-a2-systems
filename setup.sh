# Usage: source setup.sh

git config --global user.email "tomasz@biczel.com"
git config --global user.name "Tomasz Biczel"

sudo apt-get update
sudo apt-get install nsight-systems

[[ ! -e data ]] && ln -s /workspace/data/a2/ data

# To manually run QdstrmImporter on profiles.
export PATH=/usr/lib/nsight-systems/host-linux-x64:$PATH

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
# To install dependencies
uv run python

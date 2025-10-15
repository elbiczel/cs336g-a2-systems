# Usage: source setup.sh

git config --global user.email "tomasz@biczel.com"
git config --global user.name "Tomasz Biczel"
git config --global core.editor "vim"

# On lambda.ai use sudo apt, not sudo apt-get
sudo apt-get update
sudo apt-get install nsight-systems

# Enable performance events
sudo sysctl -w kernel.perf_event_paranoid=0
sudo sysctl -w kernel.kptr_restrict=0

[[ ! -e data ]] && ln -s /workspace/data/a2/ data

# To manually run QdstrmImporter on profiles.
export PATH=/usr/lib/nsight-systems/host-linux-x64:$PATH

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
# To install dependencies
uv run python

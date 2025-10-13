git config --global user.email "tomasz@biczel.com"
git config --global user.name "Tomasz Biczel"

sudo apt-get update
sudo apt-get install nsight-systems

ln -s /workspace/data/a2 data

# To install dependencies
uv run python

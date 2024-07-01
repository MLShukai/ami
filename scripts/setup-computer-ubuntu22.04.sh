# This script setups the computer (ubuntu22.04)

# Change mirror server and Upgrade
# Installs are,
# <Systems>
# ssh-server, git, tmux, nvidia-driver
# <Docker>
# docker, nvidia-container-toolkit
# <VRChat>
# obs-studio, steam
# <RemoteDesktop>
# nomachine, sunshine

# From https://linuxfan.info/ubuntu-switch-archive-mirror-command
sudo perl -p -i.bak -e 's%(deb(?:-src|)\s+)https?://(?!archive\.canonical\.com|security\.ubuntu\.com)[^\s]+%$1http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%' /etc/apt/sources.list
sudo apt update
sudo apt upgrade -y

sudo apt install openssh-server git tmux -y
sudo ubuntu-drivers autoinstall -y

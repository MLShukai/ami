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

sudo apt install -y curl openssh-server git tmux nvidia-driver-550

# --- Docker https://docs.docker.com/engine/install/ubuntu/ ---
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo docker run --rm hello-world

# No sudo
sudo groupadd docker
sudo gpasswd -a $USER docker
sudo systemctl restart docker

# Nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# --- VRChat ---
# OBS
sudo apt-get install -y ffmpeg
sudo add-apt-repository -y ppa:obsproject/obs-studio
sudo apt update
sudo apt-get install -y obs-studio

# Steam は手動でインストール。
wget https://cdn.akamai.steamstatic.com/client/installer/steam.deb
sudo dpkg -i steam.deb

# --- RemoteDesktop ---
# NoMachine
wget https://download.nomachine.com/download/8.11/Linux/nomachine_8.11.3_4_amd64.deb -O nomachine.deb
sudo dpkg -i nomachine.deb

# Sunshine
wget https://github.com/LizardByte/Sunshine/releases/download/v0.23.1/sunshine-ubuntu-22.04-amd64.deb -O sunshine.deb
sudo apt-get install -y -f ./sunshine.deb
echo 'KERNEL=="uinput", SUBSYSTEM=="misc", OPTIONS+="static_node=uinput", TAG+="uaccess"' | \
sudo tee /etc/udev/rules.d/60-sunshine.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo modprobe uinput
mkdir -p ~/.config/systemd/user
echo "[Unit]
Description=Sunshine self-hosted game stream host for Moonlight.
StartLimitIntervalSec=500
StartLimitBurst=5

[Service]
ExecStart=<see table>
Restart=on-failure
RestartSec=5s
#Flatpak Only
#ExecStop=flatpak kill dev.lizardbyte.sunshine

[Install]
WantedBy=graphical-session.target" >> ~/.config/systemd/user/sunshine.service
systemctl --user enable sunshine
systemctl --user start sunshine
systemctl --user status sunshine

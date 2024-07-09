#!/bin/bash

# This script needs to be run with root privileges
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

# Input disk device name
read -p "Enter the device name of the disk you want to mount (e.g., /dev/sdb1): " DEVICE

# Input mount point
read -p "Enter the mount point (e.g., /mnt/mydisk): " MOUNTPOINT

# Input filesystem type
read -p "Enter the filesystem type (e.g., ext4, ntfs, vfat): " FSTYPE

# Get UUID
UUID=$(blkid -s UUID -o value $DEVICE)

if [ -z "$UUID" ]; then
    echo "UUID for the specified device not found."
    exit 1
fi

# Create mount point
mkdir -p $MOUNTPOINT

# Add entry to fstab
echo "UUID=$UUID $MOUNTPOINT $FSTYPE defaults 0 2" >> /etc/fstab

# Try new mount
mount -a

if [ $? -eq 0 ]; then
    echo "Disk successfully mounted and added to /etc/fstab."
else
    echo "Mount failed. Please check /etc/fstab."
fi

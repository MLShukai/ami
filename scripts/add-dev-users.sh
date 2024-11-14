#!/bin/bash

# User List
users=("geson" "zassou" "myxy" "klutz" "bunnchinn" "suisen")

# Password Information
declare -A user_passwords

# Add User Function
add_user() {
    local username=$1

    # Generate Random Initial Password
    local initial_password=$(openssl rand -base64 12)

    # Add User
    sudo useradd -m -s /bin/bash $username

    # Set Initial Password
    echo "$username:$initial_password" | sudo chpasswd

    # Force Password Change on Next Login
    sudo passwd -e $username

    # Add User to docker group
    sudo usermod -aG docker $username

    # Create User-Specific Configuration File in sudoers.d
    echo "$username ALL=(ALL) /usr/bin/apt, /usr/bin/docker, /bin/cat" | sudo tee /etc/sudoers.d/$username

    # Set Correct Permissions for Configuration File
    sudo chmod u=r,g=r,o= /etc/sudoers.d/$username

    echo "User $username has been added and has sudo rights and docker group."

    # Save Password Information to Array
    user_passwords[$username]=$initial_password
}

# Main Process
for user in "${users[@]}"; do
    if id "$user" &>/dev/null; then
        echo "User $user already exists."
    else
        add_user $user
    fi
done

echo "User addition process completed."
echo "---------------------------------------"
echo "User name and password list:"
for user in "${!user_passwords[@]}"; do
    echo "$user : ${user_passwords[$user]}"
done
echo "---------------------------------------"
echo "Please keep these passwords safe and inform each user directly."

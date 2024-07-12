#!/usr/bin/python3

"""OBS VRChat Window Selector and Virtual Camera Starter.

Dependencies:
    - Python 3.7+
    - obs-websocket-py

Installation:
    pip install obs-websocket-py

Usage:
    1. Ensure OBS is installed and running.
    2. Enable WebSocket server in OBS (Tools -> WebSocket Server Settings).
    3. Import OBS settings:
       - Open OBS
       - Go to Scene Collection -> Import
       - Select the scene collection file from the 'obs_settings' directory
       - Go to Profile -> Import
       - Select the profile file from the 'obs_settings' directory
    4. Run this script:
       python obs_vrchat_setup.py

This script will:
    - Check if OBS is running
    - Connect to OBS via WebSocket
    - Select the VRChat window in the appropriate scene
    - Start the virtual camera if it's not already running
"""

import subprocess
import sys

from obswebsocket import exceptions, obsws, requests


def check_obs_running() -> None:
    result = subprocess.run(["pgrep", "-x", "obs"], capture_output=True, text=True)
    if result.returncode != 0:
        print("OBS is not running. Please start OBS **IN HOST DISPLAY**.")
        sys.exit(1)


def connect_obs() -> obsws:
    client = obsws("localhost", 4455, "")
    try:
        client.connect()
        return client
    except exceptions.ConnectionFailure:
        print("OBS WebSocket is not enabled or unreachable. Please check OBS settings.")
        sys.exit(1)


def select_vrchat_window(client: obsws) -> bool:
    scene_list = client.call(requests.GetSceneList()).datain
    vrchat_scene = next((scene for scene in scene_list["scenes"] if "vrchat" in scene["sceneName"].lower()), None)

    if not vrchat_scene:
        print("VRChat scene not found.")
        return False

    items = client.call(requests.GetSceneItemList(sceneName=vrchat_scene["sceneName"])).datain

    for item in items["sceneItems"]:
        source_name = item["sourceName"]
        source = client.call(requests.GetInputSettings(inputName=source_name)).datain

        if source["inputKind"] == "xcomposite_input":
            windows_list = client.call(
                requests.GetInputPropertiesListPropertyItems(inputName=source_name, propertyName="capture_window")
            ).datain

            print("Available window list:", [d["itemName"] for d in windows_list["propertyItems"]])

            vrchat_window = next((w for w in windows_list["propertyItems"] if "VRChat" == w["itemName"]), None)

            if vrchat_window:
                client.call(
                    requests.SetInputSettings(
                        inputName=source_name, inputSettings={"capture_window": vrchat_window["itemValue"]}
                    )
                )
                print(f"VRChat window selected: {vrchat_window['itemName']}")
                return True
            else:
                print("VRChat window not found in the list of available windows.")

    print("Appropriate window capture source not found.")
    return False


def start_virtual_camera(client: obsws) -> None:
    virtual_cam_status = client.call(requests.GetVirtualCamStatus()).datain
    if not virtual_cam_status["outputActive"]:
        client.call(requests.StartVirtualCam())
        print("Virtual camera started.")
    else:
        print("Virtual camera is already running.")


def main() -> None:
    check_obs_running()
    client = connect_obs()
    try:
        if select_vrchat_window(client):
            start_virtual_camera(client)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

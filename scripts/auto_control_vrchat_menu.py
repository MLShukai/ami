"""Auto Control VRChat Menu.

This script automates menu navigation in VRChat using image recognition.

Installation:
1. Install dependencies:
   pip install -r ./requirements_auto_control_vrchat_menu.txt

2. Ensure you have the necessary libraries installed.
   On Gnome Ubuntu/Debian:
   sudo apt-get install gnome-screenshot

Usage:
1. Create a 'vrchat_images' directory in the same folder as this script.
2. Create subdirectories for each scenario (e.g., 'auto_rejoin_japan_street').
3. Place required images in scenario directories, named numerically (0.png, 1.png, etc.).
4. Place 'menu.png' in the 'vrchat_images' directory or specify its location.
5. Run the script:
   python auto_control_vrchat_menu.py --scenario your_scenario_name

   Customize parameters as needed:
   python auto_control_vrchat_menu.py --window "VRChat" --scenario "auto_rejoin_japan_street" --menu_image "./vrchat_images/menu.png" --confidence 0.9 --menu_wait 1.2 --click_wait 2.0

Note: Default scenario images were created with VRChat launch options
'-screen-width 1280 -screen-height 720'. Adjust as necessary for different window sizes.
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyautogui
from Xlib import X, display
from Xlib.ext import xtest
from Xlib.XK import string_to_keysym
from Xlib.xobject.drawable import Window

# Xlib display setup
d = display.Display()
root = d.screen().root


def find_window(window_name: str) -> Optional[Window]:
    def recursive_search(window: Window) -> Optional[Window]:
        children = window.query_tree().children
        for child in children:
            name = child.get_wm_name()
            if name == window_name:
                return child
            result = recursive_search(child)
            if result is not None:
                return result
        return None

    return recursive_search(root)


def focus_window(window_name: str) -> Optional[Window]:
    window = find_window(window_name)
    if window is not None:
        window.set_input_focus(X.RevertToParent, X.CurrentTime)
        window.configure(stack_mode=X.Above)
        d.sync()
        return window
    return None


def get_all_windows() -> List[Window]:
    def recursive_get(window: Window) -> List[Window]:
        result = []
        children = window.query_tree().children
        for child in children:
            if child.get_wm_name() is not None:
                result.append(child)
            result.extend(recursive_get(child))
        return result

    return recursive_get(root)


def focus_non_vrchat_window(target_window: str) -> None:
    windows = get_all_windows()
    for window in windows:
        if window.get_wm_name() != target_window:
            window.set_input_focus(X.RevertToParent, X.CurrentTime)
            window.configure(stack_mode=X.Above)
            d.sync()
            print(f"Focused on window: {window.get_wm_name()}")
            break


def press_key(keycode: int) -> None:
    xtest.fake_input(d, X.KeyPress, keycode)
    d.sync()
    time.sleep(0.1)
    xtest.fake_input(d, X.KeyRelease, keycode)
    d.sync()


def click_at(x: int, y: int) -> None:
    xtest.fake_input(d, X.MotionNotify, x=x, y=y)
    d.sync()
    time.sleep(0.1)
    xtest.fake_input(d, X.ButtonPress, 1)
    d.sync()
    time.sleep(0.1)
    xtest.fake_input(d, X.ButtonRelease, 1)
    d.sync()


def find_image(
    image_path: str, confidence: float = 0.8, region: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[Optional[Tuple[int, int]], float]:
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    template = cv2.imread(image_path)

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > confidence:
        return ((max_loc[0] + template.shape[1] // 2, max_loc[1] + template.shape[0] // 2), max_val)
    return (None, max_val)


def find_and_click_image(
    image_path: str, confidence: float = 0.8, region: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[bool, float]:
    point, max_val = find_image(image_path, confidence, region)
    if point is not None:
        if region is not None:
            point = (region[0] + point[0], region[1] + point[1])
        click_at(point[0], point[1])
        return True, max_val
    return False, max_val


def auto_control_vrchat_menu(
    window_name: str, scenario_dir: str, menu_image: str, confidence: float, menu_wait: float, click_wait: float
) -> None:
    vrchat_window = focus_window(window_name)
    if vrchat_window is None:
        print(f"{window_name} window not found")
        return

    # Get window position and size
    geom = vrchat_window.get_geometry()
    window_region = (geom.x, geom.y, geom.width, geom.height)

    # ESC key keycode
    esc_keycode = d.keysym_to_keycode(string_to_keysym("Escape"))

    # Reset in case menu is already open
    press_key(esc_keycode)
    time.sleep(menu_wait)

    # Check if menu is open, press ESC if needed
    _, max_val = find_image(menu_image, region=window_region)
    print(f"Menu image confidence: {max_val}")
    if max_val < confidence:
        press_key(esc_keycode)
        time.sleep(menu_wait)  # Wait for menu to appear

    # Click images in sequence
    i = 0
    while True:
        image_path = os.path.join(scenario_dir, f"{i}.png")
        if not os.path.exists(image_path):
            break
        clicked, max_val = find_and_click_image(image_path, confidence=confidence, region=window_region)
        if clicked:
            print(f"Clicked on {image_path} (confidence: {max_val})")
            time.sleep(click_wait)  # Wait before next action
        else:
            print(f"Failed to find {image_path} (confidence: {max_val})")
            break
        i += 1

    # Focus on a non-VRChat window
    focus_non_vrchat_window(window_name)


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "vrchat_images")

    parser = argparse.ArgumentParser(description="Auto control VRChat menu")
    parser.add_argument("--window", type=str, default="VRChat", help="Target window name")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario directory name")
    parser.add_argument(
        "--menu_image", type=str, default=os.path.join(base_dir, "menu.png"), help="Path to the menu.png file"
    )
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold for image recognition")
    parser.add_argument("--menu_wait", type=float, default=1.0, help="Wait time after pressing ESC key")
    parser.add_argument("--click_wait", type=float, default=1.5, help="Wait time after clicking an image")

    args = parser.parse_args()

    scenario_dir = os.path.join(base_dir, args.scenario)

    if not os.path.exists(scenario_dir):
        print(f"Error: Scenario directory '{scenario_dir}' does not exist.")
        exit(1)

    if not os.path.exists(args.menu_image):
        print(f"Error: Menu image '{args.menu_image}' does not exist.")
        exit(1)

    auto_control_vrchat_menu(
        window_name=args.window,
        scenario_dir=scenario_dir,
        menu_image=args.menu_image,
        confidence=args.confidence,
        menu_wait=args.menu_wait,
        click_wait=args.click_wait,
    )

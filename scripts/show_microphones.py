#!python
import soundcard as sc

for mic in sc.all_microphones(True):
    print(f"{mic.id}: {mic.name}")

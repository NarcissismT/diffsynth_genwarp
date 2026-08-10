import time
from tqdm import tqdm
import os

inter=20000

while True:
    cmd=input('Input your cmd:\n').strip()
    if cmd!='':
        with open('/juicefs-algorithm/data/IPT/haonan_wu/Code/DiffSynth-Studio/e/cmd2exe.cmd','w')as f:
            f.write(cmd)
        for i in tqdm(range(inter),desc='Feeding...'):
            time.sleep(1/inter)

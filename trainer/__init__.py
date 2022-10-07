"""
Human in the loop deep learning segmentation for biological images

Copyright (C) 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys
from os.path import dirname
sys.path.append(dirname(__file__)) # find modules in current directory

from trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--syncdir',
                    help=('location of directory where data is'
                           ' synced between the client and server'))

def start():
    args = parser.parse_args()
    if args.syncdir:
        trainer = Trainer(sync_dir=args.syncdir)
    else:
        trainer = Trainer()
    trainer.main_loop()

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

from trainer import Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--syncdir',
                    help=('location of directory where data is'
                           ' synced between the client and server'))
parser.add_argument('--patchsize',
                    type=int,
                    default=572,
                    help=('size of patch width and height in pixels'))
parser.add_argument('--maxworkers',
                    type=int,
                    default=12,
                    help=('maximum number of workers used for the dataloader'))



if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(sync_dir=args.syncdir, patch_size=args.patchsize, max_workers=args.maxworkers)
    trainer.main_loop()

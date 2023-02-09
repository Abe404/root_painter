"""
Handle training and segmentation for a specific project

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
# pylint: disable=W0511, E1136, C0111, R0902, R0914, W0703, R0913, R0915
# W0511 is TODO
import os
import time
import warnings
import threading
from pathlib import Path
import json
import sys
from datetime import datetime
from functools import partial
import copy
import traceback
import multiprocessing

import numpy as np
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from loss import combined_loss as criterion

from datasets import TrainDataset
from metrics import get_metrics, get_metrics_str, get_metric_csv_row
from model_utils import ensemble_segment
from model_utils import create_first_model_with_random_weights
import model_utils
from model_utils import save_if_better

from im_utils import is_photo, load_image, save_then_move
from file_utils import ls
from startup import startup_setup, ensure_required_folders_exist
from unet import get_valid_patch_sizes

class Trainer():

    def __init__(self, sync_dir=None, patch_size=572, max_workers=12):

        valid_sizes = get_valid_patch_sizes()
        assert patch_size in valid_sizes, (f'Specified patch size of {patch_size}'
                f'is not valid. Valid patch sizes are {valid_sizes}')


        if sync_dir:
            self.sync_dir = sync_dir
        else:
            self.settings_path = os.path.join(Path.home(),
                                              'root_painter_settings.json')
            startup_setup(self.settings_path)
            self.sync_dir = Path(json.load(open(self.settings_path, 'r'))['sync_dir'])

        ensure_required_folders_exist(self.sync_dir)
        self.instruction_dir = os.path.join(self.sync_dir, 'instructions')
        self.training = False
        self.train_set = None
        # Can be set by instructions.
        self.train_config = None
        self.model = None
        self.first_loop = True
        self.in_w = patch_size
        self.out_w = self.in_w - 72
        mem_per_item = 3800000000
        total_mem = 0
        self.num_workers=min(multiprocessing.cpu_count(), max_workers)
        print(self.num_workers, 'workers assigned for data loader')
        print('GPU Available', torch.cuda.is_available())
        for i in range(torch.cuda.device_count()):
            total_mem += torch.cuda.get_device_properties(i).total_memory
        self.bs = total_mem // mem_per_item
        self.bs = min(12, self.bs)
        print('Batch size', self.bs)
        self.optimizer = None
        # used to check for updates
        self.annot_mtimes = []
        self.msg_dir = None
        self.epochs_without_progress = 0
        # approx 30 minutes
        self.max_epochs_without_progress = 60
        # These can be trigged by data sent from client
        self.valid_instructions = [self.start_training,
                                   self.segment,
                                   self.stop_training]

    def main_loop(self):
        print('Started main loop. Checking for instructions in',
              self.instruction_dir)
        while True:
            try:
                self.check_for_instructions()
            except Exception as e:
                print('Exception check_for_instructions', e, traceback.format_exc())
                self.log(f'Exception check_for_instructions instruction,{e},{traceback.format_exc()}')
            if self.training:
                # can take a while so checks for
                # new instructions are also made inside
                # train_one_epcoh
                self.train_one_epoch()
            else:
                self.first_loop = True
                time.sleep(1.0)

    def fix_config_paths(self, old_config):
        """ get paths relative to local machine """
        new_config = {}
        for k, v in old_config.items():
            if k == 'file_names' or k == 'format':
                # names and format specified dont need a path appending
                new_config[k] = v
            elif isinstance(v, list):
                # if its a list fix each string in the list.
                new_list = []
                for e in v:
                    new_val = e.replace('\\', '/')
                    new_val = os.path.join(self.sync_dir,
                                           os.path.normpath(new_val))
                    new_list.append(new_val)
                new_config[k] = new_list
            elif isinstance(v, str):
                v = v.replace('\\', '/')
                new_config[k] = os.path.join(self.sync_dir,
                                             os.path.normpath(v))
            else:
                new_config[k] = v
        return new_config

    def check_for_instructions(self):
        try:
            for fname in ls(self.instruction_dir):
                if self.execute_instruction(fname):
                    os.remove(os.path.join(self.instruction_dir, fname))
        except Exception as e:
            print('Exception checking for instruction', e)

    def execute_instruction(self, fname):
        fpath = os.path.join(self.instruction_dir, fname)
        name = fname.rpartition('_')[0] # remove hash
        if name in [i.__name__ for i in self.valid_instructions]:
            print('execute_instruction', name)
            try:
                with open(fpath, 'r') as json_file:
                    contents = json_file.read()
                    config = self.fix_config_paths(json.loads(contents))
                    getattr(self, name)(config)
            except Exception as e:
                print('Exception parsing instruction', e)
                print(f'{traceback.format_exc()}')
                self.log(f'Exception parsing instruction,{e},{traceback.format_exc()}')
                return False
        else:
            #TODO put in a log and display error to the user.
            raise Exception(f"unhandled instruction {name})")
        return True

    def stop_training(self, _):
        if self.training:
            self.training = False
            self.epochs_without_progress = 0
            message = 'Training stopped'
            self.write_message(message)
            self.log(message)

    def start_training(self, config):
        if not self.training:
            self.train_config = config
            self.epochs_without_progress = 0
            self.msg_dir = self.train_config['message_dir']
            model_dir = self.train_config['model_dir']
            self.train_set = TrainDataset(self.train_config['train_annot_dir'],
                                          self.train_config['dataset_dir'],
                                          self.in_w, self.out_w)
            model_paths = model_utils.get_latest_model_paths(model_dir, 1)
            if model_paths:
                self.model = model_utils.load_model(model_paths[0])
            else:
                self.model = create_first_model_with_random_weights(model_dir)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                                             momentum=0.99, nesterov=True)
            self.model.train()
            self.training = True

    def reset_progress_if_annots_changed(self):
        train_annot_dir = self.train_config['train_annot_dir']
        val_annot_dir = self.train_config['val_annot_dir']
        new_annot_mtimes = []
        for annot_dir in [train_annot_dir, val_annot_dir]:
            for fname in ls(annot_dir):
                fpath = os.path.join(annot_dir, fname)
                new_annot_mtimes.append(os.path.getmtime(fpath))
        new_annot_mtimes = sorted(new_annot_mtimes)
        if new_annot_mtimes != self.annot_mtimes:
            print('reset epochs without progress as annotations have changed')
            self.epochs_without_progress = 0
        self.annot_mtimes = new_annot_mtimes

    def write_message(self, message):
        """ write a message for the user (client) """
        Path(os.path.join(self.msg_dir, message)).touch()

    def train_one_epoch(self):
        train_annot_dir = self.train_config['train_annot_dir']
        val_annot_dir = self.train_config['val_annot_dir']
        if not [is_photo(a) for a in ls(train_annot_dir)]:
            return
        if not [is_photo(a) for a in ls(val_annot_dir)]:
            return

        if self.first_loop:
            self.first_loop = False
            self.write_message('Training started')
            self.log('Starting Training')

        train_loader = DataLoader(self.train_set, self.bs, shuffle=True,
                                  # 12 workers is good for performance
                                  # on 2 RTX2080 Tis (but depends on CPU also)
                                  # 0 workers is good for debugging
                                  # don't go above max_workers (user specified but default 12) 
                                  # and don't go above the number of cpus, provided by cpu_count.
                                  num_workers=self.num_workers,
                                  drop_last=False, pin_memory=True)
        epoch_start = time.time()
        self.model.train()
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        defined_total = 0
        loss_sum = 0
        for step, (photo_tiles,
                   foreground_tiles,
                   defined_tiles) in enumerate(train_loader):

            self.check_for_instructions()
            photo_tiles = photo_tiles.cuda()
            foreground_tiles = foreground_tiles.cuda()
            defined_tiles = defined_tiles.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(photo_tiles)
            softmaxed = softmax(outputs, 1)
            # just the foreground probability.
            foreground_probs = softmaxed[:, 1, :]
            # remove any of the predictions for which we don't have ground truth
            # Set outputs to 0 where annotation undefined so that
            # The network can predict whatever it wants without any penalty.
            outputs[:, 0] *= defined_tiles
            outputs[:, 1] *= defined_tiles
            loss = criterion(outputs, foreground_tiles)
            loss.backward()
            self.optimizer.step()
            foreground_probs *= defined_tiles
            predicted = foreground_probs > 0.5

            # we only want to calculate metrics on the
            # part of the predictions for which annotations are defined
            # so remove all predictions and foreground labels where
            # we didn't have any annotation.

            defined_list = defined_tiles.view(-1)
            preds_list = predicted.view(-1)[defined_list > 0]
            foregrounds_list = foreground_tiles.view(-1)[defined_list > 0]

            # # calculate all the false positives, false negatives etc
            tps += torch.sum((foregrounds_list == 1) * (preds_list == 1)).cpu().numpy()
            tns += torch.sum((foregrounds_list == 0) * (preds_list == 0)).cpu().numpy()
            fps += torch.sum((foregrounds_list == 0) * (preds_list == 1)).cpu().numpy()
            fns += torch.sum((foregrounds_list == 1) * (preds_list == 0)).cpu().numpy()
            defined_total += torch.sum(defined_list > 0).cpu().numpy()
            loss_sum += loss.item() # float
            sys.stdout.write(f"Training {(step+1) * self.bs}/"
                             f"{len(train_loader.dataset)} "
                             f" loss={round(loss.item(), 3)} \r")
            self.check_for_instructions() # could update training parameter
            if not self.training:
                return

        duration = round(time.time() - epoch_start, 3)
        print('epoch train duration', duration)
        self.log_metrics('train', get_metrics(tps, fps, tns, fns,
                                              defined_total, duration))
        before_val_time = time.time()
        self.validation()
        print('epoch validation duration', time.time() - before_val_time)

    def log_metrics(self, name, metrics):
        fname = datetime.today().strftime('%Y-%m-%d')
        fname += f'_{name}.csv'
        fpath = os.path.join(self.train_config['log_dir'], fname)
        if not os.path.isfile(fpath):
            # write headers if file didn't exist
            print('date_time,true_positives,false_positives,true_negatives,'
                  'false_negatives,precision,recall,f1,defined,duration',
                  file=open(fpath, 'w+'))
        with open(fpath, 'a+') as log_file:
            log_file.write(get_metric_csv_row(metrics))
            log_file.flush()

    def validation(self):
        """ Get validation set metrics for current model and previous model.
             log those metrics and update the model if the
             current model is better than the previous model.
             Also stop training if the current model hasnt
             beat the previous model for {max_epochs}
        """
        model_dir = self.train_config['model_dir']
        # TODO consider implementing checkpointer class to maintain
        # this state.
        get_val_metrics = partial(model_utils.get_val_metrics,
                                  val_annot_dir=self.train_config['val_annot_dir'],
                                  dataset_dir=self.train_config['dataset_dir'],
                                  in_w=self.in_w, out_w=self.out_w, bs=self.bs)
        prev_model, prev_path = model_utils.get_prev_model(model_dir)
        cur_metrics = get_val_metrics(copy.deepcopy(self.model))
        prev_metrics = get_val_metrics(prev_model)
        self.log_metrics('cur_val', cur_metrics)
        self.log_metrics('prev_val', prev_metrics)
        was_saved = save_if_better(model_dir, self.model, prev_path,
                                   cur_metrics['f1'], prev_metrics['f1'])
        if was_saved:
            self.epochs_without_progress = 0
        else:
            self.epochs_without_progress += 1

        self.reset_progress_if_annots_changed()

        message = (f'Training {self.epochs_without_progress}'
                   f' of max {self.max_epochs_without_progress}'
                   ' epochs without progress')
        print(message)
        self.write_message(message)
        if self.epochs_without_progress >= self.max_epochs_without_progress:
            message = (f'Training finished as {self.epochs_without_progress}'
                       ' epochs without progress')
            print(message)
            self.log(message)
            self.training = False
            self.write_message(message)

    def write_train_metrics(self, metrics):
        metric_str = get_metrics_str(metrics,
                                     to_use=['f1_score', 'recall',
                                             'precision', 'duration'])
        message = 'Training-' + metric_str
        print(message)
        self.log(message)
        self.write_message(message)

    def log(self, message):
        with open(os.path.join(self.sync_dir, 'server_log.txt'), 'a+') as log_file:
            log_file.write(f"{datetime.now()}|{time.time()}|{message}\n")
            log_file.flush()

    def segment(self, segment_config):
        """
        Segment {file_names} from {dataset_dir} using {model_paths}
        and save to {seg_dir}.

        If model paths are not specified then use
        the latest model in {model_dir}.

        If no models are in {model_dir} then create a
        random weights model and use that.

        TODO: model saving is a counter-intuitve side effect,
        re-think project creation process to avoid this
        """
        in_dir = segment_config['dataset_dir']
        seg_dir = segment_config['seg_dir']
        format_str = 'RootPainter Default (.png)'
        if 'format' in segment_config:
            format_str = segment_config['format']
        
        if "file_names" in segment_config:
            fnames = segment_config['file_names']
        else:
            # default to using all files in the directory if file_names is not specified.
            fnames = ls(in_dir)

        # if model paths not specified use latest.
        if "model_paths" in segment_config:
            model_paths = segment_config['model_paths']
        else:
            model_dir = segment_config['model_dir']
            model_paths = model_utils.get_latest_model_paths(model_dir, 1)
            # if latest is not found then create a model with random weights
            # and use that.
            if not model_paths:
                create_first_model_with_random_weights(model_dir)
                model_paths = model_utils.get_latest_model_paths(model_dir, 1)
        start = time.time()
        for fname in fnames:
            self.segment_file(in_dir, seg_dir, fname,
                              model_paths, format_str, sync_save=len(fnames) == 1)
        duration = time.time() - start
        print(f'Seconds to segment {len(fnames)} images: ', round(duration, 3))
        
    def segment_file(self, in_dir, seg_dir, fname, model_paths, format_str, sync_save):
        fpath = os.path.join(in_dir, fname)

        # When the client navigates through images, there is a risk that 
        # they may not realise that training has not been started.
        # These segmentation instructions keep getting processed so
        # use this as an opportunity to let them know the network is
        # not training
        try: 
            if not self.training:
                # if the seg dir is in the same folder as a folder called messages.
                # then assume messages should go to this folder.
                proj_dir = os.path.dirname(seg_dir)
                msg_dir = os.path.join(proj_dir, 'messages')
                if os.path.isdir(msg_dir) and not self.msg_dir:
                    self.msg_dir = msg_dir
                if self.msg_dir:
                    message = "Network not training"
                    self.write_message(message)
                else:
                    # if we didn't find an obvious message location
                    # then the current instruction might not be assocated with
                    # any particular project, so do not send a message.
                    pass

        except Exception as e:
            stack = traceback.format_exc()
            print('excpetion writing mesage', e, stack)



        npy = False
        if format_str == 'Numpy Compressed (.npz)':
            # segmentation output is a binary map.
            npy = True
            out_path = os.path.join(seg_dir, os.path.splitext(fname)[0] + '.npz')
        else:
            out_path = os.path.join(seg_dir, os.path.splitext(fname)[0] + '.png')

        if os.path.isfile(out_path):
            print('Skip because found existing segmentation file')
            return
        if not os.path.isfile(fpath):
            print('Cannot segment as missing file', fpath)
        else:
            try:
                photo = load_image(fpath)
            except Exception as e:
                # Could be temporary issues reading the image.
                # its ok just skip it.
                print('Exception loading', fpath, e)
                return
            seg_start = time.time()
            seg_out = ensemble_segment(model_paths, photo, self.bs,
                                         self.in_w, self.out_w)
            print(f'ensemble segment {fname}, dur', round(time.time() - seg_start, 2))
            # catch warnings as low contrast is ok here.
            with warnings.catch_warnings():
            
                # create a version with alpha channel
                warnings.simplefilter("ignore")

                if format_str == 'RhizoVision Explorer (.png)':
                    # RVE needs segmentation in black and white
                    # Load RootPainter blue channel and invert.
                    seg_out = (seg_out == 0)
                elif npy:
                    seg_out = seg_out.astype(bool)
                else:
                    # default output is PNG with alpha channel
                    seg_alpha = np.zeros((seg_out.shape[0], seg_out.shape[1], 4))
                    seg_alpha[seg_out > 0] = [0, 1.0, 1.0, 0.7]
                    # Conver to uint8 to save as png without warning
                    seg_out  = (seg_alpha * 255).astype(np.uint8)


                if sync_save:
                    # other wise do sync because we don't want to delete the segment
                    # instruction too early.
                    save_then_move(out_path, seg_out, npy)
                else:
                    # TODO find a cleaner way to do this.
                    # if more than one file then optimize speed over stability.
                    x = threading.Thread(target=save_then_move,
                                         args=(out_path, seg_out, npy))
                    x.start()

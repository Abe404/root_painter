import psutil
from threading import Thread
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


class IntDataset(Dataset):

    def __init__(self):
        print('init dataset')

    def __len__(self):
        return 612*10

    def __getitem__(self, idx):
        time.sleep(1)
        #return [np.random.random((3, 572, 572)).astype(np.float32),
        #        np.random.random((3, 572, 572)).astype(np.int),
        #        np.random.random((3, 572, 572)).astype(np.int)]
        return idx

def loader_worker():
    global shuffle_buffer
    global epoch_updates
    global loader_fin
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8)
    for i, item in enumerate(dataloader):
        shuffle_buffer.append([0, item])
        
        if len(shuffle_buffer) > shuffle_buffer_limit:
            # remove the oldest item
            oldest_item = shuffle_buffer[0]
            for s in shuffle_buffer:
                if s[0] > oldest_item[0]:
                    oldest_item = s
            shuffle_buffer.remove(oldest_item)
        if (epoch_updates*batch_size) > 612:
            break # break out but leave the updater running until we get out of this loop
    loader_fin = True # now we are out of the loop tell updater to stop.

def trainer_worker():
    global shuffle_buffer 
    global epoch_updates
    global loader_fin
    start = time.time()

    while True:
        if len(shuffle_buffer) > 0:
            item = shuffle_buffer[0]
            for s in shuffle_buffer:
                # compare age
                if s[0] < item[0]:
                    item = s

            #shuffle_buffer[shuffle_buffer.index(item)][0] += 1
            item[0] += 1
            epoch_updates += 1
            time_since = time.time() - start
            print('time since last update', round(time.time() - start, 2),
                  'buffer len = ', len(shuffle_buffer),
                  'update count', epoch_updates, 
                  end='\r')
            start = time.time()
            time.sleep(0.4)
        # keep going until the data loader thread has finished.
        # it takes a while for it to shut down (seconds)
        # so we may as well keep updating the network with the shuffle buffer
        if loader_fin:
            print('updater hit lim')
            print('buffer len = ', len(shuffle_buffer), 'update count', epoch_updates)
            print('')
            return
        if (epoch_updates*batch_size) > (612 * 2):
            # We don't want this to go on forever, even if something goes wrong.
            print('we have gone way over the limit. assume data loader crashed'
                  ' and that its time to finisih the epoch anyway') 
            return

def epoch(name):
    global epoch_updates, loader_fin

    epoch_updates = 0
    # flag used to inform trainer when the data loader has finished.
    loader_fin = False 
    epoch_start = time.time()
    loader_thread = Thread(target=loader_worker)
    trainer_thread = Thread(target=trainer_worker)
    loader_thread.start()
    trainer_thread.start()
    loader_thread.join()
    trainer_thread.join()
    print(f'epoch {name} time = ', round(time.time() - epoch_start, 2))
    print('min item train count = ', min([s[0] for s in shuffle_buffer]))
    print('max item train count = ', max([s[0] for s in shuffle_buffer]))
    print('mean train count = ', np.mean([s[0] for s in shuffle_buffer]))

epoch_updates = 0
shuffle_buffer = []
dataset = IntDataset()
batch_size = 4
shuffle_buffer_limit = 128 // batch_size # uses around 10GB of shared memory
loader_fin = False


if __name__ == '__main__':
    epoch('one')
    epoch('two')
    epoch('three')

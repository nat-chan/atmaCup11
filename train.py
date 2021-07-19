#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import *
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.make_deterministic import make_deterministic
from trainers.pix2pix_trainer import Pix2PixTrainer
from tqdm import tqdm
from pathlib import Path


# modelの初期重みをdeterministicにする epoch数を避けて10000とする
make_deterministic(seed=10000)

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataset = data.find_dataset_using_name(opt.dataset_mode)()
dataset.initialize(opt)
print(f"dataset {type(dataset).__name__} of size {len(dataset)} was created")
if not len(dataset)%opt.batchSize == 0:
    print("\x1b[31;1mYOU CAN NOT RESUME TRAIN DETERMINISTICALLY! CHECK BATCHSIZE!\x1b[m")


# create trainer for our model
trainer = Pix2PixTrainer(opt)

#XXX with open("/home/natsuki/hoge.log","a") as f:
#XXX     log = str(trainer.pix2pix_model.netG.model.conv1.weight).split("\n")[1]
#XXX     f.write(f"{__file__} {log}\n")

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataset))

# create tool for visualization
#TODO visualizer = Visualizer(opt)

loss_log = open(Path(opt.checkpoints_dir)/opt.name/"loss.log", "w")

#TEST loaded: List[str] = list()
for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    skip = iter_counter.total_steps_so_far % len(dataset)
    iter_counter.epoch_iter = skip
    dataset.shuffle(seed=epoch) # dataの列をdeterministicにshuffleする
    dataloader = data.partial_dataloader(opt, dataset, range(skip, len(dataset)))
    pbar = tqdm(dataloader, dynamic_ncols=True, initial=skip//opt.batchSize, total=len(dataset)//opt.batchSize)
    for i, data_i in enumerate(pbar, start=iter_counter.epoch_iter//opt.batchSize):
        #TEST loaded += data_i['path']
        iter_counter.record_one_iteration()

        trainer.run_generator_one_step(data_i)

        losses = trainer.get_latest_losses()
        MSE: float = losses["MSE"].mean().item()
        print(iter_counter.total_steps_so_far, MSE, file=loss_log, flush=True)
        pbar.set_description(f'epoch={epoch} skip={skip//opt.batchSize} total={iter_counter.total_steps_so_far} MSE={str(MSE)[:5]}')

        # Visualizations
#        if iter_counter.needs_printing():
#TODO            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far) #

#        if iter_counter.needs_displaying():
#            if not (opt.leak_low == -1 and opt.leak_high == -1):
#                visuals = OrderedDict([('input_lineart', 1 - 2*data_i['hed']),
#                                        ('input_hint', (data_i['image']-1)*data_i['mask']+1),
#                                        ('synthesized_image', trainer.get_latest_generated()),
#                                        ('real_image', data_i['image'])])
#            else:
#                visuals = OrderedDict([('input_lineart', 1 - 2*data_i['hed']),
#                                        ('synthesized_image', trainer.get_latest_generated()),
#                                        ('real_image', data_i['image'])])
#            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print(f'saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})')
            if opt.save_steps:
                trainer.save('s%s' % iter_counter.total_steps_so_far)
                trainer.save('latest')
            else:
                trainer.save('latest')
            iter_counter.record_current_iter()
    pbar.close()
    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(f'saving the model at end of (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})')
        trainer.save('latest')
        trainer.save(epoch)

loss_log.close()
print('Training was successfully finished.')

#TEST shuffleが決定的かlogを吐いてdiffを取ったら完全一致した！
#TEST from time import time
#TEST now = int(time())
#TEST with open(f"{now}.log", "w") as f:
#TEST         txt = "\n".join(loaded)
#TEST         f.write(txt)
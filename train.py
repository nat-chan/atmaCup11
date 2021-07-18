#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from tqdm import tqdm

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataset = data.find_dataset_using_name(opt.dataset_mode)()
dataset.initialize(opt)
print(len(dataset))
print(f"dataset {type(dataset).__name__} of size {len(dataset)} was created")


# create trainer for our model
# trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataset))

# create tool for visualization
# visualizer = Visualizer(opt) TODO

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    skip = iter_counter.total_steps_so_far % len(dataset)
    iter_counter.epoch_iter = skip
    dataloader = data.partial_dataloader(opt, dataset, range(skip, len(dataset)))
    pbar = tqdm(dataloader, dynamic_ncols=True, initial=skip//opt.batchSize, total=len(dataset)//opt.batchSize)
    for i, data_i in enumerate(pbar, start=iter_counter.epoch_iter//opt.batchSize):
#        print(data_i['path'][0].split('/')[-1][:-4])
        iter_counter.record_one_iteration()
        pbar.set_description(f'epoch={epoch} skip={skip//opt.batchSize} total={iter_counter.total_steps_so_far}')

        # Training
        # train generator
#TODO        if i % opt.D_steps_per_G == 0:
#TODO            trainer.run_generator_one_step(data_i)

        # train discriminator
#TODO        trainer.run_discriminator_one_step(data_i)

        # Visualizations
#        if iter_counter.needs_printing(): TODO visual
#            losses = trainer.get_latest_losses()
#            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far) #

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
#TODO            if opt.save_steps:
#TODO                trainer.save('s%s' % iter_counter.total_steps_so_far)
#TODO                trainer.save('latest')
#TODO            else:
#TODO                trainer.save('latest')
            iter_counter.record_current_iter()
    pbar.close()
#TODO    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(f'saving the model at end of (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})')
#TODO        trainer.save('latest')
#TODO        trainer.save(epoch)

print('Training was successfully finished.')

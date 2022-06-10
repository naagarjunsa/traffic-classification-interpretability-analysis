from nam.config import defaults
from nam.data import FoldedDataset, NAMDataset
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np


metadata = pd.read_csv("active_device_case/metadata.csv")
print("Number of samples : ", len(metadata))
print("Number of target classes : ", len(set(metadata.iloc[:, 1])))

samples, features, labels = [], [], []
flag = True

count = 1

for _, row in metadata.iterrows():
    fname = "active_device_case/" + row[0].split(".")[0] + ".npt"
    file_info = pd.read_csv(fname)

    if flag:
        feature_list = file_info.columns.values.tolist()
        for i in range(21):
            features.extend(
                [str(i) + "_" + feature for feature in feature_list])
        flag = False

    curr_sample = []
    packet_count = 0
    for _, file_row in file_info.sample(1).iterrows(): # taking only 1 packet
        curr_sample.extend(file_row)
        packet_count = packet_count + 1

    for i in range(21-packet_count):
      curr_sample.extend([-1]*len(feature_list))
    samples.append(curr_sample)
    labels.append(row[1].split("-")[0])
    count = count +1

df_samples = pd.DataFrame(samples, columns=features)

df_labels = pd.DataFrame(labels, columns=["label"])
df_data = df_samples.join(df_labels, how="outer")

print("Number of samples : ", len(samples))
print("Number of features per sample : ", len(samples[0]), len(features))
print("Number of labels : ", df_labels['label'].nunique())
print("Unique of labels : ", df_labels['label'].unique())
df_data.fillna(-1)
df_data.head()

df_data['label'] = df_data['label'].map({'ubiquoss':0,'axis':1,'mikrotik':2,'cisco':3,'avtech' :4, 'zte':5, 'roku' :6,'lancom' :7,'h3c' :8, 'huawei':9, 'juniper':10, 'chromecast' :11, 'adtran':12, 'nec':13, 'dell':14})

df_data.fillna(-1)
df = df_data.isnull().any()
df1 = pd.DataFrame(data=df.index, columns=['feature'])
df2 = pd.DataFrame(data=df.values, columns=['bool'])
df_any = pd.merge(df1, df2, left_index=True, right_index=True)

df_any['bool'].sum()

df_data = df_data.drop(columns=['0_src_ip', '1_src_ip', '2_src_ip', '3_src_ip', '4_src_ip', '5_src_ip', '6_src_ip', '7_src_ip', '8_src_ip', '9_src_ip', '10_src_ip', '11_src_ip', '12_src_ip',
                       '13_src_ip', '14_src_ip', '15_src_ip', '16_src_ip', '17_src_ip', '18_src_ip', '19_src_ip', '20_src_ip', 'label'], axis=1)

df_data.sample(10)

config = defaults()
print(config)

config.num_epochs = 20
config.batch_size = 64
config.device = "cpu"
df_data = NAMDataset(config, data_path=df_data,
                     features_columns=df_data.columns[:-1], targets_column=df_data.columns[-1])
dataloaders = df_data.train_dataloaders()

model = NAM(config=config, name='NAM_DEVICE_CLASS', num_inputs=len(
    df_data[0][0]), num_units=get_num_units(config, df_data.features),)
print(model)

for fold, (trainloader, valloader) in enumerate(dataloaders):

    tb_logger = TensorBoardLogger(
        save_dir=config.logdir, name=f'{model.name}', version=f'fold_{fold + 1}')

    checkpoint_callback = ModelCheckpoint(
        filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}", monitor='val_loss', save_top_k=config.save_top_k, mode='min')

    litmodel = LitNAM(config, model)
    trainer = pl.Trainer(logger=tb_logger, max_epochs=config.num_epochs,
                         enable_checkpointing=checkpoint_callback, accelerator="cpu")

    trainer.fit(litmodel, train_dataloaders=trainloader,
                val_dataloaders=valloader)

trainer.test(litmodel, dataloaders=df_data.test_dataloaders())

fig1 = plot_mean_feature_importance(litmodel.model, df_data)

ax = fig1.get_axes()

print(ax[-1].get_xticklabels()[-10:])

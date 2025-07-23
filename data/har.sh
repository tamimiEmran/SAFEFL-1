#!/bin/bash
# download and unzip dataset
wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip -O UCI_HAR_Dataset.zip

unzip UCI_HAR_Dataset.zip

mv "UCI HAR Dataset" HAR

rm -rf __MACOSX

rm UCI_HAR_Dataset.zip
#!/bin/bash

# Prepare virtualenv
source /home/jccarr/.env/bin/activate


dset_name="z7-8.5_HII-DIM-128_BOX_LEN-128"

printf "\n\n-----\t Start of $dset_name\t-----\n" >> test.log

# python3 <filepath-to-main.py> \
#         <directory-to-save-data> \
#         <dataset-name> \
#         <path-to-config-file> \
#         <function to run>

python3 coeval-simulator/bin/main.py \
        "/home/jccarr/scratch/datasets" \
        $dset_name \
        coeval-simulator/in/config_default.yml \
        --make_coeval_dset

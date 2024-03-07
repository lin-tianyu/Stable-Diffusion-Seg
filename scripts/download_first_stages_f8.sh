#!/bin/bash
wget -O models/first_stage_models/kl-f8/model.zip https://ommer-lab.com/files/latent-diffusion/kl-f8.zip

cd models/first_stage_models/kl-f8
unzip -o model.zip


cd ../..
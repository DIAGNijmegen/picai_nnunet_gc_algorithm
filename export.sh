#!/usr/bin/env bash

./build.sh

docker save picai_baseline_nnunet_processor | gzip -c > picai_baseline_nnunet_processor_1.0.tar.gz

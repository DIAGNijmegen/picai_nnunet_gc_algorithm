#!/usr/bin/env bash

./build.sh

docker save picai_baseline_nnunet_processor | gzip -c > picai_baseline_nnunet_processor_2.1.tar.gz

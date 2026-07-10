#! /bin/bash

METHOD_TYPE="adaptive_srkl"
METHOD_NAME="distillm"
USE_CREL=1
USE_STUDENT_GEN=1

source "$(dirname "${BASH_SOURCE[0]}")/common_train.inc" "$@"

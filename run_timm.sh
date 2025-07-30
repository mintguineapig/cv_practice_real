#!/bin/bash
MODEL_NAME=$1
shift

python - "$@" <<PYCODE
import sys, runpy, timm, models

# 몽키패치: img_size=32 옵션 추가
models.ResNet18 = lambda num_classes: timm.create_model(
    "${MODEL_NAME}",
    pretrained=False,
    num_classes=num_classes,
    img_size=32
)

sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path("main.py", run_name="__main__")
PYCODE

## generate train.txt and test.txt

1. copy data to this directory
2. run `python gen_my_train_test_data.py` in terminal

## train

run `python my_detector.py --phase train` in terminal

## finetune

run `python my_detector.py --phase finetune --load-model [modelPath]` in terminal
Training base on the given model

## predict

run `python my_detector.py --phase predict --input <inputPath> --out <outPath>` in terminal

## more information

logs 
#!/bin/bash 

echo 'Testing with base weights:' $1 
echo 'Save name:' $2 
echo 'Dataset:' $3 


# echo Testing data
# if [ $3 == coco ]
# then
#    python3 test_data.py --subset train --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
# else 
#    python3 test_data.py --subset train07 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
#    python3 test_data.py --subset train12 --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
    
# fi

python3 test_data.py --subset val --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3
python3 test_data.py --subset test --dir "${1}${NUM}" --saveNm "${2}${NUM}" --dataset $3

echo Associating data
python3 associate_data.py FRCNN --saveNm "${2}" --dataset $3

echo Getting Results
python3 get_results.py FRCNN --saveNm "${2}" --dataset $3 --saveResults True

#!/usr/bin/env sh

DATA="/home/tim/datasets/cifar10"

cd $DATA

rm -f train.txt
rm -f test.txt

find train/0 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 0/" >> train.txt
find train/1 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 1/" >> train.txt
find train/2 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 2/" >> train.txt
find train/3 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 3/" >> train.txt
find train/4 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 4/" >> train.txt
find train/5 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 5/" >> train.txt
find train/6 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 6/" >> train.txt
find train/7 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 7/" >> train.txt
find train/8 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 8/" >> train.txt
find train/9 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 9/" >> train.txt


find test/0 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 0/" >> test.txt
find test/1 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 1/" >> test.txt
find test/2 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 2/" >> test.txt
find test/3 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 3/" >> test.txt
find test/4 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 4/" >> test.txt
find test/5 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 5/" >> test.txt
find test/6 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 6/" >> test.txt
find test/7 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 7/" >> test.txt
find test/8 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 8/" >> test.txt
find test/9 -name "*" | grep -i -E ".bmp|.jpg|.png" | sed "s/$/ 9/" >> test.txt

#!/bin/bash
# 有错误，还得改，先放个样板在这
for Filename in $(ls -l /etc |awk '$5 > 10240 {print $9}')
do
cp $Filename /tmp
done
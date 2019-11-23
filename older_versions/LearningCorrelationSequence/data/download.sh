#!/bin/bash 
download(){ 
if [ ! -f $1 ]; then 
wget http://skuld.cs.umass.edu/traces/storage/$1.spc.bz2 -O $1.spc.bz2 
bzip2 -d $1.spc.bz2
mv $1.spc $1.csv
fi 
} 
download Financial1
download Financial2
download WebSearch1
download WebSearch2
download WebSearch3

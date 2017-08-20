#!/bin/bash

OUTPUT=$( wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p' )
CODE=${OUTPUT##*Code: }
echo $CODE

F='GoogleNews-vectors-negative300.bin.gz'
if [ -z "$1" ]
        then
                OUT_F=$F
        else
                OUT_F=$1/$F
fi
echo $OUT_F

wget --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&confirm='$CODE'&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O $OUT_F

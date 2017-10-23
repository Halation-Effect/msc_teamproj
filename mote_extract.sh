#!/bin/bash

awk 'BEGIN {OFS=",";} $4 ~ /^9$/ {print $1, $2, $5, $6, $7}' data.txt > mote_extract.csv


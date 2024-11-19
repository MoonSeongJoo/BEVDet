#!/bin/bash

# download bevdet from github
tag=dev2.0
url=https://github.com/HuangJunJie2017/BEVDet/

wget -c $url/archive/$tag.zip
unzip $tag.zip
rm dev2.0.zip

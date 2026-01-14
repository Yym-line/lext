#!/bin/bash

folder="/node/yym/node/dataset/kws/mix-chunk/farrever5"

folder_name=$(basename "$folder")

output_dir="/node/yym/node/dataset/kws/mix-chunk"
output="$output_dir/$folder_name.scp"


> "$output"  # 清空输出文件

find "$folder" -type f | while read -r file; do
    filename=$(basename "$file")
    echo "$filename $file" >> "$output"
done


#!/bin/bash -e
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Measures the size of an xtensa binary by parsing the output of
# xt-size. If an optional list of symbols is provided, the symbols' sizes
# are excluded from the total. This is useful when the binary contains symbols
# that are only used during testing.
#
# First argument is the binary location.
# Second argument is a regular expression for symbols that need to be excluded
# from the measurement

declare -r TEST_TMPDIR=/tmp/test_xtensa_binary/
declare -r MICRO_LOG_PATH=${TEST_TMPDIR}/$1
declare -r MICRO_LOG_FILENAME=${MICRO_LOG_PATH}/logs.txt
mkdir -p ${MICRO_LOG_PATH}

cp $1 $1.elf

raw_size=$(xt-size $1.elf)
# Skip the title row
sizes=$(echo "${raw_size}" | sed -n '2 p')
text_size=$(echo "$sizes" | awk '{print $1}')
data_size=$(echo "$sizes" | awk '{print $2}')
bss_size=$(echo "$sizes" | awk '{print $3}')
total_size=$(echo "$sizes" | awk '{print $4}')

symbols=$(xt-nm -S $1.elf | grep -w $2)

while IFS= read -r line; do
  symbol_size=$((16#$(echo $line | awk '{print $2}')))
  symbol_type=$(echo $line | awk '{print $3}')
  symbol_name=$(echo $line | awk '{print $4}')

  total_size=$(("$total_size"-"$symbol_size"))
  if [[ "$symbol_type" =~ [DdRr] ]]; then
    # Data and readonly symbols are counted as data
    data_size=$(("$data_size"-"$symbol_size"))
  elif [[ "$symbol_type" =~ [Tt] ]]; then
    # Text symbols
    text_size=$(("$text_size"-"$symbol_size"))
  elif [[ "$symbol_type" =~ [Bb] ]]; then
    # BSS symbols
    bss_size=$(("$bss_size"-"$symbol_size"))
  else
    echo "The symbol $(symbol_name)'s type isn't recognized"
    exit 1
  fi
done <<< "$symbols"
str="text   data   bss   total
$text_size $data_size $bss_size $total_size"
echo "$str"
exit 0

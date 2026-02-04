#!/bin/bash
# split the wav scp, calculate duration and merge
nj=4
. tools/parse_options.sh || exit 1;

inscp=$1
outscp=$2
data=$(dirname ${inscp})
if [ $# -eq 3 ]; then
  logdir=$3
else
  logdir=${data}/log
fi
mkdir -p ${logdir}

rm -f $logdir/wav_*.slice
rm -f $logdir/wav_*.shape
rm -f $logdir/wav_*.log

# 兼容 macOS 的 split 命令
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS BSD split: 使用 -l 按行数分割
    total_lines=$(wc -l < $inscp)
    lines_per_file=$(( (total_lines + nj - 1) / nj ))  # 向上取整
    split -l $lines_per_file -a 2 $inscp $logdir/wav_
    # 重命名文件添加 .slice 后缀 (只重命名 split 产生的文件)
    for file in $logdir/wav_[a-z][a-z]; do
        if [ -f "$file" ]; then
            mv "$file" "${file}.slice"
        fi
    done
else
    # Linux GNU split
    split --additional-suffix .slice -d -n l/$nj $inscp $logdir/wav_
fi

for slice in `ls $logdir/wav_*.slice`; do
{
    name=`basename -s .slice $slice`
    tools/wav2dur.py $slice $logdir/$name.shape 1>$logdir/$name.log
} &
done
wait
cat $logdir/wav_*.shape > $outscp

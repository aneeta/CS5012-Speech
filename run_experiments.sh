#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Pass target directory name in which to store result files."
    exit
fi

# make a folder for the experiment files
datadir=$1
figdir=figs
[ -d "$datadir" ] || mkdir -p "$datadir"
[ -d "$datadir/$figdir" ] || mkdir -p "$datadir/$figdir"

output_file=${datadir}/combined_results.csv
log_file=$datadir/log.txt

# varied dimensions
# all supported languages run by default
smoothing=("WB" "GT")
unking=(0 1)

for s in "${smoothing[@]}"
do
    for u in "${unking[@]}"
    do
        # add unk tags if true, else don't
        [[ $u = 1 ]] && unk="--unk" || unk=""
        ref=S-${s}_UNK-${u}
        filename=${datadir}/res_${ref}.csv
        figname=${datadir}/${figdir}/${ref}
        echo "Running for $ref"
        python p1.py --smoothing $s $unk \
                     --plot $figname --csv $filename &>> $log_file
    done
done

# Combine all CSVs
# make a header
awk -F"," 'BEGIN { OFS = "," } NR == 1 { print $0, "smoothing", "unk" ; next }' "$filename" > "$output_file"

for file in "$datadir"/res_*.csv
do
    offset=`expr length $datadir`
    s=${file:`expr ${offset} + 7`:2}
    u=${file:`expr ${offset} + 14`:1}
    [[ $u = 1 ]] && unk="True" || unk="False"
    # copy over all the data
    awk -F"," -v v1="$s" -v v2="$unk" 'BEGIN { OFS = "," } NR > 1 { print $0, v1, v2 ; next }' "$file" >> "$output_file"
    # clean up
    rm $file
done

echo "Completed"

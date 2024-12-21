
langs=('de-en' 'en-fr' 'en-cs' 'en-hi' 'ro-en')

for lang in ${langs[@]}; do
  IFS='-'
  read -a pair <<< $lang
  echo $lang
  SRC=${pair[0]}
  TGT=${pair[1]}
  IFS=' '
  lang=$SRC-$TGT
  data_path='data/'$lang

  python PMI-Align.py --srcAdd $data_path/text.$SRC --tgtAdd $data_path/text.$TGT --alignFile $data_path/gold.$lang.aligned \
  --layers 0 24 --outAdd outputs/$lang/xlmr-large --model "xlm-roberta-large" --oneRef > Results/$lang.PMI.XLMR-large

done

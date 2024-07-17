# sh new_lr_sqm_13.sh 2 0.9 0
device=$1
split_size=$2
seed=$3
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=8192 mu_normalized=2.09 device=$device q=0.001 epoch=2 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=8192 mu_normalized=1.354 device=$device q=0.001 epoch=5 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=8192 mu_normalized=0.944 device=$device q=0.001 epoch=8 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=8192 mu_normalized=0.71 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=CA is_onehot=True gamma=8192 mu_normalized=0.525 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed

python3 lr_sqm.py data_type=folktable_income folk_state=TX is_onehot=True gamma=8192 mu_normalized=2.09 device=$device q=0.001 epoch=2 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=TX is_onehot=True gamma=8192 mu_normalized=1.354 device=$device q=0.001 epoch=5 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=TX is_onehot=True gamma=8192 mu_normalized=0.944 device=$device q=0.001 epoch=8 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=TX is_onehot=True gamma=8192 mu_normalized=0.71 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=TX is_onehot=True gamma=8192 mu_normalized=0.525 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed

python3 lr_sqm.py data_type=folktable_income folk_state=NY is_onehot=True gamma=8192 mu_normalized=2.09 device=$device q=0.001 epoch=2 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=NY is_onehot=True gamma=8192 mu_normalized=1.354 device=$device q=0.001 epoch=5 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=NY is_onehot=True gamma=8192 mu_normalized=0.944 device=$device q=0.001 epoch=8 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=NY is_onehot=True gamma=8192 mu_normalized=0.71 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=NY is_onehot=True gamma=8192 mu_normalized=0.525 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed

python3 lr_sqm.py data_type=folktable_income folk_state=FL is_onehot=True gamma=8192 mu_normalized=2.09 device=$device q=0.001 epoch=2 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=FL is_onehot=True gamma=8192 mu_normalized=1.354 device=$device q=0.001 epoch=5 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=FL is_onehot=True gamma=8192 mu_normalized=0.944 device=$device q=0.001 epoch=8 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=FL is_onehot=True gamma=8192 mu_normalized=0.71 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed
python3 lr_sqm.py data_type=folktable_income folk_state=FL is_onehot=True gamma=8192 mu_normalized=0.525 device=$device q=0.001 epoch=10 split_size=$split_size seed=$seed

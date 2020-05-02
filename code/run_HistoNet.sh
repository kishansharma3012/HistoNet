#Prepare data 
python data_utils.py --data_dir="/HistoNet/data/FlyLarvae_dataset/" --target_dir="/HistoNet/data/"

# Train and evaluate network
THEANO_FLAGS=device=cpu python main.py --dataset_path="/HistoNet/data/Train_val_test/" --experiment_name="HistoNet_01_05_2020" --output_dir="/HistoNet/Result_10_5/" --loss_name="w_L1" --num_epochs=2 --batch_size=2 --num_bins="8" --Loss_wt="0.5,0.5" --lr_decay=0.95 --lr=8e-3 --reg=1e-4 --DSN="False"

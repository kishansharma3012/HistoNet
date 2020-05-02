# HistoNet: Predicting size histograms of object instances
A new data set of pixel-wise instance labeled soldier fly larvaes.

**Dataset**-[Download](https://github.com/kishansharma3012/FlyLarvae_dataset/blob/master/FlyLarvae_dataset.zip)

**Paper**-[Link](http://openaccess.thecvf.com/content_WACV_2020/html/Sharma_HistoNet_Predicting_size_histograms_of_object_instances_WACV_2020_paper.html)

## Dataset
### Description
This dataset of soldier fly larvae, which are bred in massive quantities for sustainable, environment friendly organic waste decomposition [1,2]. Fly larvae images were collected using a Sony Cyber shot DSC-WX350 camera with image size 1380 X 925 pixels The camera is installed on a professional fixture to guarantee a fixed distance from camera to observation plane for all image acquisitions. This is important to avoid any scale variation between the different image acquisitions. Very large numbers of larvae mingled with a lot of larvae feed lead to high object overlap and occlusions. These similar looking brown colored fly larvaes have variations in their sizes and flexible structure. To simplify our tasks, we choose high contrast black color background. The images were collected by using one spoonful of larvaes weighting approximately 3-6 grams (~700-1900 Larvaes), uniformly scattered over the image area. All larvae instances are labeled pixel-wise.

<img src="https://github.com/kishansharma3012/FlyLarvae_dataset/blob/master/fixture_sample.png">

### Properties
This dataset represent crowded scenario of similar looking fly larvaes. The size (pixel area covere) distribution histogram of fly larvae is shown in figure. This dataset consists of 10844 pixel-wise labeled fly larvaes. The average size of fly larvae is 120.2 +- 28.1 px 
<img src="https://github.com/kishansharma3012/FlyLarvae_dataset/blob/master/size_histogram.png">

To get the individual labeled instance and total number of objects in an image use Scikit-image library's measure label function. To mesaure the size, centroid and other information of individual object use measure.regionprops function from Scikit-image library.

Note: the given mask labels have 4 different pixel values (50, 140, 200, 255), a single object have same pixel value, and the object near to each other are represented using different pixel value. To get different pixel values for individual object use Scikit-library's measure label function.

### Data-preparation
step1: Sampling 256x256 images
step2: Data-augmentation
step3: Calculating target size histogram and redundant object countmap.
```bash
python data_utils.py --data_dir="path_to_HistoNet/data/FlyLarvae_dataset/" --target_dir="path_to_HistoNet/data/"
```
## Training HistoNet
### Environment
Create environment:
```bash
 conda create --name <env> --file req.txt
```
```bash
python main.py --dataset_path="path_to_HistoNet/data/Train_val_test/" --experiment_name="HistoNet_01_05_2020" --output_dir="path_to_HistoNet/Result_10_5/" --loss_name="w_L1" --num_epochs=2 --batch_size=2 --num_bins="8" --Loss_wt="0.5,0.5" --lr_decay=0.95 --lr=8e-3 --reg=1e-4 --DSN="False"
```

## References
1. S. Diener, N.M.S. Solano, F.R. Guti´errez, C. Zurbr ¨ ugg, and K. Tockner. Biological treatment of municipal organic waste using black soldier fly larvae. Waste and Biomass Valorization, 2(4):357–363, 2011.
2. M. Gold, J.K. Tomberlin, S. Diener S., C. Zurbr ¨ ugg, and A. Mathys. Decomposition of biowaste macronutrients, microbes, and chemicals in black soldier fly larval treatment: A review. Waste and Biomass Valorization, 82:302–318, 2018.

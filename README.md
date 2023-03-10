# prostate_t2map
Code to develop and evaluate NNs for parameter estimation

This archive contains all the code and models used in the following manuscript:
Improved Quantitative Parameter Estimation for Prostate T2 Relaxometry using Convolutional Neural Networks
[CITATION TBD]

You can use this code to generate the synthesized data, train the models, run inference, and create all the figures in the paper. However, this process is not fully automated - you will need to do this in parts, with some manual steps. You can use the pre-trained models, or retrain them youself with a few days of compute time. 


1) Configure python. You can use the requirements_basic2_20230103.txt file to create a python virtual environment matching mine. I ran all of the code for this paper from the Spyder developmennt environment.


2) Download ImageNet data. From this website: 
https://www.kaggle.com/datasets/samfc10/ilsvrc2012-validation-set
Download the archive with 50,000 files to your system. On my system they are stored in /mnt/data/ReferenceDatasets/imagenet_sm. 


3) Update paths. In the python file utility_functions.py I have several hardwired paths, either relative to my "base" path (/home/pbolan/prostate_t2map), or pointing to a location with the downloaded imagenet data. You'll want to update these for your system.


4) Download and organize the in vivo data. 
Under the base directory create a "datasets" directory. The publicly available files are available here: https://conservancy.umn.edu/handle/11299/243192
as a single file, images.tgz. Expand this, and you'll find imagesTr (training) and imagesTs (testing). You'll first move these into folders to create three 3D datasets (set1 = development, set2=training, set3=testing). 

Create a folder called $BASE/datasets/invivo_set1/images/, and copy the first nine training files (prostate_00*.nii.gz) from imagesTr into there. This is optional.
Create a folder called $BASE/datasets/invivo_set2/images/, and move all the *.nii.gz files from imagesTr into there. 
Create a folder called $BASE/datasets/invivo_set3/images/, and move all the *.nii.gz files from imagesTs into there. 

All of those files are 3D niftis, and they need to be extracted into 2D slices. To do this, run the script extract_slices_from_invivo_datasets.py. 


3) Generate synthetic data. Run build_synthetic_datasets.py to build the testing and training datasets. Not I have a big (10k) training set, and a smaller one (1k) that I used for development. 


4) Train models. Run train_1d.py and train_cnn.py. These will take a few days, depending on your system, and will write out all the models into $BASE/models/*.pt. You can skip this and just use the pre-trained models available on github.


5) Perform inference.
In the file inference.py, the inference is broken into 3 parts:
part A: synthetic data. 
part B: invivo
part C: invivo with noise addition. 
You can run inference.py to do all of them, or just run the parts you want. 
This will create the predicted values in the folder $BASE/predictions, about 7GB


6) Run analyses and generate plots. 
This is more manual, and done in several steps. While developing I often used inline graphics (%matplotlib inline), but for the paper I used Qt as the renderer, saving png and svg files and arranging them into figures manually. 
make_demo_figure.py will make the plots for figure 1. You'll need to run several times, uncommenting lines to make all sections.
plot_example_partA.py will make the plots for figure 3
make_plots_partA.py will do all per-slice and per-pixel evaluations on synthetic data, figures 4 and 5. Will take a few hours for the per-pixel stuff. PR
plot_example_partB.py will make plots for figure 6
comparison_matrix.py will make figure 7
evaluate_partC_byslice.py will make figure 8



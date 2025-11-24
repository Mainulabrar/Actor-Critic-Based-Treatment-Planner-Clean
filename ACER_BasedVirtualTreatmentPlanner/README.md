Contents:

This folder contains the code for virtual treatment planners for cancer radiotherpay treatment planner along with folders that contains supporting code and data.

=================To set the environemnt

Run 

conda env create -f environment.yml

=================To train:

To run the training and validation of the treatment planner, run the TrainingAndValidation.py code.

=================To test:

To test the treatment planner on dataset 1 with 7 beam configuration, run TestingDataset1SevenBeamAngles.py 

To test the treatment planner on dataset 1 with 6 beam configuration, run TestingDataset1SixBeamAngles.py 

To test the treatment planner on dataset 1 with 7 beam configuration with random treatment planning parameter initialization, run TestingDataset1randomizedInitializationSevenBeams.py

To test the treatment planner on dataset 2 with 7 beam configuration, run TestingDataset2.py 

To test the treatment planner on dataset 3 with 25 beam configuration, run TestingDataset3.py 

=================To conduct an first gradient sign menthod based adversarial attack:

Run  FGSM_Attack001CrossEntropyDataSet2.py


# Introduction

**Default configuration**:

* Data root: "/home/cotai/giang/datasets/VOC-2012"

# Lecture outline

## Coding lecture schedule

There are totally 9 coding sessions. Each session lasts for 2 - 3 hours.

* Session 1: Explore VOC-2012 dataset for segmantic segmentation

* Session 2: Write torch Dataset for VOC-2012 dataset

* Session 3: Write trainer class for training segmentation model on VOC-2012 dataset

* Session 4: Write model class for training segmentation model on VOC-2012 dataset

* Session 5: Write optimizers, learning rate scheduler, loss functions, and metrics for training and evaluation

* Session 6: Build an end-to-end training pipeline for semantic segmentation

* Session 7: Write evaluation script and visualization code for VOC-2012 dataset

* Session 8: Play around with models, loss functions, metrics, optimizers, and schedulers (2)

* Session 9: Hold a small inner-class competition based on VOC-2012 dataset

## Lectures

* Session 1: Introduction to Semantic Segmentation

    * Theory: Introduction to  semantic segmentation and VOC-2012 dataset
    
    * Coding: Explore VOC-2012 dataset with Python

* Session 2: Implement torch Dataset for VOC-2012 dataset

* Session 3: Popular segmentation models
    
    * Theory: 
        
        * UNet and its relatives (e.g. FCN, LinkNet, etc.)
        
        * DeeplabV3 and its relatives (e.g. PSPNet, etc.)
    
    * Coding: Implement UNet with PyTorch

* Topic 4: Loss functions for semantic segmentation

    * Theory: 
        
        * Cross-Entropy loss, Weighted Cross-Entropy loss, Focal loss
        
        * DICE loss, HD loss, soft IoU loss
    
    * Coding: Implement loss functions in PyTorch

* Topic 5: Building a Trainer for semantic segmentation

    * Theory: 
        
        * Commonly used optimizers and schedulers in semantic segmentation
        
        * Commonly used training pipeline for semantic segmentation model
    
    * Coding: 
        
        * Implement optimizers and schedulers for semantic segmentation
        
        * Build a simple training pipeline for semantic segmentation, i.e. a `Trainer` class including
        
            * `__get_model`, `__get_optimizer`, `__get_dataloaders` methods
        
            * `train_one_epoch` method
            
            * `eval_one_epoch` method

* Topic 6: Class imbalance problem in VOC-2012 segmentation datasset, and metrics to evaluate semantic segmentation model
    
    * Theory:
        
        * Symptoms of class imbalance - high IoU scores and low loss values
        
        * IoU, Precision, and Recall
    
    * Coding:
        
        * Implement metric functions to evaluate semantic segmentation model
        
        * Integrate metric functions into the trainer class
        
        * Refine the trainer class for better visualization and best-model saving

* Topic 7: Visualization and debugging
    
    * Theory:
        
        * Review about semantic segmentation and training-and-evaluation pipeline
    
    * Coding:
        
        * Visualize model predictions with Python
        
        * Debug semantic segmentation model

* Topic 8: Play around with semantic segmentation
    
    * Theory:
        
        * Brain-storm session about semantic segmentation modeling
        
        * Brain-storm session about loss functions for semantic segmentation
    
    * Coding:
        
        * Wrap trainer class to flexibly try out different models
        
        * Try different semantic segmentation models
        
        * Try different loss functions for semantic segmentation

* Topic 9: Hold a class-level competition about semantic segmentation with VOC-2012 dataset
    
    * Rules:
        
        * Students brain-storm or even implement their ideas, e.g. models, loss functions, etc. beforehand at home
        
        * Students continue working on the competition at the class
        
        * The last 1 - 2 hours, depending on the number of students / teams, will be dedicated for evaluation and solution presentation
### Skin Lesion Classifier

#### Objective:

Build a classifier model that can distinguish between 3 classes of skin disorders.

#### Methodology:

#### Dataset:

Initial set:

3 classes:

Acne (n= 40)

Herpes simplex (n= 16)

Lichen planus (n= 56)

Total n= 112

Source:  http://www.danderm-pdv.is.kkh.dk/atlas/index.html  (with permission obtained by OROHealth)

Test set:

A separate test dataset was created (3 images per class).

Source:  https://www.pcds.org.uk/clinical-a-z-list

#### Data Preparation:

1.  The images were verified.  One image could not be opened and was removed from the set.
2.  The images were converted to tensors.
3.  The PyTorch/ Fastai module DataBlock was utilized to parse the image with its label, perform the train/validation set split (80:20), perform resizing and transformations.

#### Modelling:

A simple CNN was constructed




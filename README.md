This project contains basic experiments on german traffic signs ([the GTSRB dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)) 
recognition using linear SVMs, PHOW features and Homogeneous Kernel Mapping.

To try out this experimental code run `download.sh` which downloads dataset, download vlfeat with `download_vlfeat.sh` script 
and compile the code with `make`. After that execute `run.sh` which will train a classifier and apply it to test data outputting
the accuracy.

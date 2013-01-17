echo 'Downloading training images'
#wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
unzip GTSRB_Final_Training_Images.zip -d .
echo 'Downloading test images'
#wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_Images.zip -d .
echo 'Downloading extra annotations for test images'
#wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip
unzip GTSRB_Final_Test_GT.zip -d GTSRB/Final_Test/Images

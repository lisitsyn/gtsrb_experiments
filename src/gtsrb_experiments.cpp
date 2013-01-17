#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/preprocessor/SumOne.h>
#include <shogun/preprocessor/HomogeneousKernelMap.h>
#include "phow_features.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdio.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <bitset>
#include <dirent.h>
#include <algorithm>

using namespace cv; 
using namespace shogun;
using namespace std;

#define RESIZE_TO 36

Mat load_file(const string& filename)
{
	return imread(filename.c_str(),false);
}

vector<string>& split(const string &s, char delim, vector<string>& elems) 
{
	elems.clear();
	stringstream ss(s);
	string item;
	while(getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

Mat transform_image(Mat image,int RoiX1, int RoiY1, int RoiX2, int RoiY2, int xj=0, int yj=0, int s=0) 
{
	int X1 = RoiX1+xj;
	int X2 = RoiX2-xj;
	int Y1 = RoiY1+yj;
	int Y2 = RoiY2-yj;
	Mat roi = Mat(image,Rect(X1,Y1,X2-X1+1,Y2-Y1+1));
	Mat roi_colorspace = roi.clone();
	Mat roi_resized;
	resize(roi,roi_resized,Size(RESIZE_TO,RESIZE_TO),0.0,0.0,INTER_AREA);
	Mat roi_norm = roi_resized.clone();
	equalizeHist(roi_resized,roi_norm);
	
	return roi_norm;
}

vector< pair<Mat, int> > read_test()
{
	int total_count = 0;
	vector< pair<Mat, int> > test_data;
	string dir = "GTSRB/Final_Test/Images/";
	ifstream file((dir + "/GT-final_test.csv").c_str());
	string line;
	vector<string> tokens;
	Mat image;
	Mat image_transformed;
	if (file.is_open())
	{
		getline(file,line);
		int c = 0;
		while (file.good())
		{
			getline(file,line);
			if (line.empty())
				continue;
			split(line,';',tokens);
			string filename = tokens[0];
			int RoiX1, RoiY1, RoiX2, RoiY2;
			stringstream(tokens[3]) >> RoiX1;
			stringstream(tokens[4]) >> RoiY1;
			stringstream(tokens[5]) >> RoiX2;
			stringstream(tokens[6]) >> RoiY2;
			int width,height;
			stringstream(tokens[1]) >> width;
			stringstream(tokens[2]) >> height;
			image = load_file(dir+filename);
			image_transformed = transform_image(image,RoiX1,RoiY1,RoiX2,RoiY2);
			int ClassId;
			stringstream(tokens[7]) >> ClassId;
			test_data.push_back(pair<Mat, int>(image_transformed,ClassId));
			c++;
		}
		total_count += c;
		file.close();
	}
	cout << "Total test count: " << total_count << endl;
	return test_data;
}


vector< pair<Mat, int> > read_training()
{
	const int n_classes = 43;
	int total_count = 0;
	vector< pair<Mat, int> > train_data;
	for (int i=0; i<n_classes; i++)
	{
		string dir = "GTSRB/Final_Training/Images/";
		stringstream numeric_stream;
		numeric_stream << setfill('0') << setw(5) << i;
		string numeric = numeric_stream.str();
		ifstream file((dir + numeric + "/GT-" + numeric + ".csv").c_str());
		cout << "reading " << dir+numeric+"/GT-"+numeric+".csv" << endl;
		string line;
		vector<string> tokens;
		Mat image;
		Mat image_transformed1;
		Mat image_transformed2;
		Mat image_transformed3;
		Mat roi_resized;
		Mat roi_norm;
		if (file.is_open())
		{
			getline(file,line);
			int c = 0;
			//while (c<50)
			while (file.good())
			{
				getline(file,line);
				if (line.empty())
					continue;
				split(line,';',tokens);
				string filename = tokens[0];
				int width,height;
				stringstream(tokens[1]) >> width;
				stringstream(tokens[2]) >> height;
				int RoiX1, RoiY1, RoiX2, RoiY2;
				stringstream(tokens[3]) >> RoiX1;
				stringstream(tokens[4]) >> RoiY1;
				stringstream(tokens[5]) >> RoiX2;
				stringstream(tokens[6]) >> RoiY2;
				image = load_file(dir+numeric+"/"+filename);
				image_transformed1 = transform_image(image,RoiX1,RoiY1,RoiX2,RoiY2,0,0);
				int ClassId;
				stringstream(tokens[7]) >> ClassId;
				train_data.push_back(pair<Mat, int>(image_transformed1,ClassId));
				c++;
			}
			total_count += c;
			file.close();
		}
	}
	cout << "Total training count: " << total_count << endl;
	return train_data;
}

int main(int argc, const char** argv)
{
	/// show_var();
	vector<pair<Mat,int> > train_data = read_training();
	vector<pair<Mat,int> > test_data = read_test();
	init_shogun_with_defaults();
	get_global_io()->enable_progress();

	/// PHOW
	vector<pair<SGVector<float64_t>,int> > phow_train_descriptors = phow_compute_features(train_data);
	vector<pair<SGVector<float64_t>,int> > phow_test_descriptors = phow_compute_features(test_data);
	CDenseFeatures<float64_t>* phow_train_features = new CDenseFeatures<float64_t>();
	CMulticlassLabels* phow_train_labels = new CMulticlassLabels(phow_train_descriptors.size());
	SG_REF(phow_train_features); SG_REF(phow_train_labels);
	phow_fill_features(phow_train_features,phow_train_labels,phow_train_descriptors,true);
	CDenseFeatures<float64_t>* phow_test_features = new CDenseFeatures<float64_t>();
	CMulticlassLabels* phow_test_labels = new CMulticlassLabels(phow_test_descriptors.size());
	SG_REF(phow_test_features); SG_REF(phow_test_labels);
	phow_fill_features(phow_test_features,phow_test_labels,phow_test_descriptors,true);

	CDotFeatures* features = phow_train_features;
	CDotFeatures* test_features = phow_test_features;
	CMulticlassLabels* labels = phow_train_labels;
	CMulticlassLabels* test_labels = phow_test_labels;

	CSumOne* norm_l1 = new CSumOne();
	norm_l1->apply_to_feature_matrix(features);
	norm_l1->apply_to_feature_matrix(test_features);
	SG_UNREF(norm_l1);

	CHomogeneousKernelMap* hkm = 
		new CHomogeneousKernelMap(HomogeneousKernelIntersection,HomogeneousKernelMapWindowRectangular);
	hkm->apply_to_feature_matrix(features);
	hkm->apply_to_feature_matrix(test_features);
	SG_UNREF(hkm);

	CNormOne* norm_l2 = new CNormOne();
	norm_l2->apply_to_feature_matrix(features);
	norm_l2->apply_to_feature_matrix(test_features);
	SG_UNREF(norm_l2);

	CMulticlassLibLinear* classifier = new CMulticlassLibLinear(1.25,features,labels);
	classifier->io->set_loglevel(MSG_DEBUG);
	classifier->set_epsilon(1e-3);
	classifier->set_max_iter(3000);
	classifier->train(features);
	CMulticlassLabels* predict = classifier->apply_multiclass(test_features);
	CMulticlassAccuracy* acc = new CMulticlassAccuracy();
	cout << "Accuracy " << acc->evaluate(predict, test_labels) << endl;

	SG_UNREF(classifier);
	SG_UNREF(predict);
	exit_shogun();
	return 1;
}

#include <bitset>
#include <vl/dsift.h>
#include <shogun/lib/SGVector.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
using namespace shogun;

vector<pair<SGVector<float64_t>,int> > phow_compute_features(vector<pair<Mat,int> > data, vector<int> pairs=vector<int>())
{
	vector< Mat > images;
	vector< vector<KeyPoint> > keypoints;
	vector< SGVector<float64_t> > descriptors;

	int size = data[0].first.cols;
	VlDsiftFilter* dsift = vl_dsift_new_basic(size,size,8,4);
	float* img = (float*)malloc(sizeof(float)*size*size);
	vl_dsift_set_flat_window(dsift,true);
	for (unsigned int i=0; i<data.size(); i++)
	{
		Mat& image_ = data[i].first;
		for (int q=0; q<size; q++)
		{
			for (int p=0; p<size; p++)
				img[q*size+p] = image_.at<uint8_t>(q,p)/255.0;
		}
		vl_dsift_process(dsift,img);
		SGVector<float64_t> dscr(vl_dsift_get_keypoint_num(dsift)*vl_dsift_get_descriptor_size(dsift));
		float const* vl_dscr = vl_dsift_get_descriptors(dsift);
		for (int q=0; q<dscr.size(); q++)
		{
			dscr[q] = vl_dscr[q];
		}
		descriptors.push_back(dscr);
		images.push_back(image_);
		SG_SPROGRESS(i,0,data.size());
	}
	free(img);
	vl_dsift_delete(dsift);
	vector<pair<SGVector<float64_t>, int> > res;
	for (unsigned int i=0; i<data.size(); i++)
	{
		res.push_back(pair<SGVector<float64_t>, int>(descriptors[i],data[i].second));
	}
	return res;
}

void phow_fill_features(CDenseFeatures<float64_t>* features, CMulticlassLabels* labels, vector<pair<SGVector<float64_t>, int> > descriptors, bool bitwise)
{
	int n_dims = descriptors[0].first.size();
	features->set_feature_matrix(SGMatrix<float64_t>(n_dims,descriptors.size()));
	for (unsigned int i=0; i<descriptors.size(); i++)
	{
		features->set_feature_vector(descriptors[i].first,i);
		labels->set_label(i,descriptors[i].second);
	}
}

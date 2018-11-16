/*
 *  Copyright 2018 Digital Media Professionals Inc.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include <iostream>
#include <string>
#include <limits>
#include <opencv2/opencv.hpp>

#include "CaffeGoogLeNet_gen.h"
#include "imagenet_1000_categories.h"

using namespace std;

static const size_t NET_INPUT_HEIGHT = 224;
static const size_t NET_INPUT_WIDTH  = 224;

void usage(void)
{
	cout << "Usage: ./main images..." <<endl;
}

// This preprocessing do the belows
// - convert uint8_t to __fp16
// - transpose image
// - normalize image
void preproc_image(const uint8_t *src, __fp16 *dst, size_t width, size_t height)
{
	for(size_t c = 0; c < width; c++){
		for(size_t r = 0; r < height; r++){
			dst[(c * height + r) * 3 + 0] = (__fp16)(src[(r * width + c) * 3 + 0] - 128);
			dst[(c * height + r) * 3 + 1] = (__fp16)(src[(r * width + c) * 3 + 1] - 128);
			dst[(c * height + r) * 3 + 2] = (__fp16)(src[(r * width + c) * 3 + 2] - 128);
		}
	}
}

int read_and_preprocess_image(const string &path, __fp16 *input_buf,
							  			size_t width, size_t height)
{
	cv::Mat raw;
	cv::Mat rsz;
	cv::Mat rgb;

	// read image
	raw = cv::imread(path);
	if(raw.data == NULL){
		cerr << "fail to load an image from " << path << endl;
		return -1;
	}
	cv::resize(raw, rsz, cv::Size(width, height));
	cv::cvtColor(rsz, rgb, cv::COLOR_BGR2RGB);
	if(!rgb.isContinuous()){
		rgb = rgb.clone();
	}

	// preproc_image
	preproc_image(rgb.data, input_buf, width, height);

	return 0;
}


int init_net(CCaffeGoogLeNet &net, void **input_addr)
{
	if(!net.Initialize()){
		cerr << "fail to initialize network" << endl;
		return -1;
	}
	if(!net.LoadWeights("CaffeGoogLeNet/CaffeGoogLeNet_weights.bin")){
		cerr << "fail to load weight" << endl;
		return -1;
	}
	if(!net.Commit()){
		cerr << "fail to commit network" << endl;
		return -1;
	}
	*input_addr = net.get_network_input_addr_cpu();
	return 0;
}

template <typename T>
int argmax(vector<T> &v)
{
	T max = numeric_limits<T>::min();
	int max_i = -1;
	for(unsigned i = 0; i < v.size(); i++){
		if(max < v[i]){
			max_i = i;
			max = v[i];
		}
	}
	return max_i;
}

int main(int argc, const char *argv[])
{
	int ret = 0;
	CCaffeGoogLeNet net;
	void *net_input_addr = nullptr;

	// usage
	if(argc <= 1){
		usage();
		return -1;
	}

	// init
	if(init_net(net, &net_input_addr) != 0){
		cerr << "fail to initialize network" << endl;
		return -1;
	}

	// main loop
	for(int i = 1; i < argc && ret == 0; i++){
		__fp16 input_buf[NET_INPUT_WIDTH * NET_INPUT_HEIGHT * 3];
		const char *img_path = argv[i];
		std::vector<float> output;

		// read and preprocess image
		if(read_and_preprocess_image(img_path, input_buf, NET_INPUT_WIDTH, NET_INPUT_HEIGHT) < 0){
			ret = -1;
			break;
		}
		
		// input image and run network
		memcpy(net_input_addr, input_buf, NET_INPUT_WIDTH * NET_INPUT_HEIGHT * 6);
		if(!net.RunNetwork()){
			cerr << "fail to run network" << endl;
			ret = -1;
			break;
		}

		// get and output result
		net.get_final_output(output);
		cout << img_path << " , " << categories[argmax(output)] << endl;
	}

	return ret;
}

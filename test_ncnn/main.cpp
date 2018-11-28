#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "net.h"
#include <opencv2/opencv.hpp>

int main()
{
    std::string param_path = "model/small.param.bin";
    std::string bin_path = "model/small.bin";
    std::string file_path ="data/test.jpg";

	std::string prama_bin = "model/small_all.bin";
    const int target_size = 224;
    auto frame = cv::imread(file_path);
	if(frame.rows > frame.cols){
		cv::copyMakeBorder(frame, frame, 0,0,0, frame.rows - frame.cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
	} else{
		cv::copyMakeBorder(frame, frame, 0,frame.cols - frame.rows,0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
	}
	cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows, target_size, target_size);
	const float mean_vals[3] = {0.f, 0.f, 0.f};
	const float norm_vals[3] = {1.f/255, 1.f/255, 1.f/255};
	in.substract_mean_normalize(mean_vals, norm_vals);
    // ncnn model and parameters
    ncnn::Net model;
    FILE* fp = fopen(prama_bin.c_str(), "rb");
	std::cout<<"radf"<<std::endl;
    model.load_param_bin(fp);
    model.load_model(fp);
    fclose(fp);

    std::cout<<"radf"<<std::endl;
//
//    model.load_param_bin(param_path.c_str());
//    model.load_model(bin_path.c_str());


	const float mean_vals_1[3] = {0.485f, 0.456f, 0.406f};
	const float norm_vals_1[3] = {1/0.229f, 1/0.224f, 1/0.225f};
	in.substract_mean_normalize(mean_vals_1, norm_vals_1);

    // ncnn output
    ncnn::Extractor extractor = model.create_extractor();
    extractor.set_light_mode(true);
    extractor.input(0, in);
    ncnn::Mat out;
    extractor.extract(0, out);

    std::cout<< out.h << " w: "<<out.w<<std::endl;
	for(int i = 1; i < 80 + 1; ++i){
		std::cout<< out[i-1]<<" ";
		if(i%8 == 0) std::cout<<std::endl;
	}

    std::cout << "Every thiing done" << std::endl;
    return 0;
}


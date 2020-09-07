//
// Created by Vitaliy Vorobyov on 25.08.2020.
//

#include <iostream>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/features2d/features2d.hpp"

#include "CommandLineParser.h"

namespace po = boost::program_options;

imageStitcher::CommandLineParser::CommandLineParser(int argc, char** argv) {
    this->argc = argc;
    this->argv = argv;
}

int imageStitcher::CommandLineParser::ParseImages(std::vector<cv::Mat>& images) {
    po::options_description desc("General options");
    desc.add_options()
            ("help,h", "Show help")
            ("images,i", po::value<std::vector<std::string>>()->multitoken(), "Input images for stitching")
            ;

    po::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return EXIT_SUCCESS;
    }
    else if (vm.count("images"))
    {
        std::vector<std::string> image_names = vm["images"].as<std::vector<std::string>>();
        for (auto & image_name : image_names) {
            cv::Mat img = cv::imread(cv::samples::findFile(image_name));
            if (img.empty()) {
                std::cout << "Can't read image '" << image_name << std::endl;
                return EXIT_FAILURE;
            }
            images.push_back(std::move(img));
        }
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "No images were provided" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <boost/program_options.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace boost::program_options;

int main(int argc, char* argv[])
{
    options_description desc("General options");
    desc.add_options()
        ("help,h", "Show help")
        ("images,i", value<vector<string>>()->multitoken(), "Input images for stitching")
    ;

    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if (vm.count("help")) {
        cout << desc << endl;
        return EXIT_SUCCESS;
    }

    if (vm.count("images")) {
        vector<Mat> images;

        vector<string> image_names = vm["images"].as<vector<string>>();
        for (int i = 0; i < image_names.size(); ++i) {
            Mat img = imread(samples::findFile(image_names[i]));
            if (img.empty()) {
                cout << "Can't read image '" << image_names[i] << endl;
                return EXIT_FAILURE;
            }
            images.push_back(img);
        }

        vector<vector<KeyPoint>> keypoints;
        vector<Mat> descriptors;

        Ptr<BRISK> brisk = BRISK::create();
        for (int i = 0; i < images.size(); ++i) {
            vector<KeyPoint> keypoint;
            Mat descriptor;
            brisk->detectAndCompute(images[i], Mat(), keypoint, descriptor);
            keypoints.push_back(keypoint);
            descriptors.push_back(descriptor);
        }

        BFMatcher bfMatcher(NORM_HAMMING, true);

        std::vector<vector<DMatch>> descriptorsMatches;
        for (int i = 0; i < descriptors.size() - 1; ++i) {
            std::vector<DMatch> match;
            bfMatcher.match(descriptors[i], descriptors[i + 1], match);
            descriptorsMatches.push_back(match);
        }

        auto sortRuleLambda = [] (DMatch const& s1, DMatch const& s2) -> bool
        {
            return s1.distance < s2.distance;
        };

        for (int i = 0; i < descriptorsMatches.size(); ++i) {
            sort(descriptorsMatches[i].begin(), descriptorsMatches[i].end(), sortRuleLambda);
        }

        vector<vector<Point2f>> matchedSrc, matchedDst;
        for (int i = 0; i < descriptorsMatches.size(); ++i) {
            vector<Point2f> src,dst;
            for (int j = 0; j < descriptorsMatches[i].size()/2; j++)
            {
                src.push_back(keypoints[i][descriptorsMatches[i][j].queryIdx].pt+Point2f(0,images[i].rows));
                dst.push_back(keypoints[i+1][descriptorsMatches[i][j].trainIdx].pt);
            }
            matchedSrc.push_back(src);
            matchedDst.push_back(dst);
        }

        Mat result;
        for (int i = 0; i < descriptorsMatches.size(); ++i) {
            Mat h=findHomography(matchedSrc[i],matchedDst[i],RANSAC);

            warpPerspective(images[i+1], result, h.inv(), Size(2*images[i+1].cols +images[i].cols , 2*images[i+1].rows+images[i].rows));

            Mat roi1(result, Rect(0, images[i].rows, images[i].cols, images[i].rows));
            images[i].copyTo(roi1);
        }
        imwrite("result.jpg",result);
    } else {
        cout << "No images were provided" << endl;
        return EXIT_FAILURE;
    }
}

Mat stitch_image(Mat image1, Mat image2, Mat H) {
    cv::Mat result;
    warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
    cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
    image2.copyTo(half);
}
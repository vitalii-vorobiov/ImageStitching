#include <opencv2/opencv.hpp>

#include "CommandLineParser.h"
#include "ImageStitcher.h"

auto sortRuleLambda = [] (cv::DMatch const& s1, cv::DMatch const& s2) -> bool
{
    return s1.distance < s2.distance;
};

int main(int argc, char** argv)
{
    imageStitcher::CommandLineParser commandLineParser(argc, argv);
    imageStitcher::ImageStitcher imageStitcher;

    std::vector<cv::Mat> images;
    commandLineParser.ParseImages(images);

    cv::Mat result = images[0];

    for (int i = 1; i < images.size(); ++i)
    {
        std::vector<cv::Mat> img;
        img.push_back(std::move(result));
        img.push_back(std::move(images[i]));

        std::vector<std::vector<cv::KeyPoint>> keypoints;
        std::vector<cv::Mat> descriptors;
        imageStitcher.keypointsAndDescriptors(img, keypoints, descriptors);

        cv::BFMatcher bfMatcher(cv::NORM_HAMMING, true);
        std::vector<std::vector<cv::DMatch>> descriptorsMatches;
        imageStitcher.matchDescriptors(descriptorsMatches, bfMatcher, keypoints, descriptors);

        for (int i = 0; i < descriptorsMatches.size(); ++i)
        {
            sort(descriptorsMatches[i].begin(), descriptorsMatches[i].end(), sortRuleLambda);
        }

        std::vector<std::vector<cv::Point2f>> matchedSrc, matchedDst;
        imageStitcher.matchedSrcDst(matchedSrc, matchedDst, descriptorsMatches, keypoints, img);

        imageStitcher.combineImages(descriptorsMatches, matchedSrc, matchedDst, img, result);
    }
    cv::imwrite("result.jpg",result);
}
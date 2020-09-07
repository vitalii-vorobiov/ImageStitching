//
// Created by Vitaliy Vorobyov on 25.08.2020.
//

#ifndef IMAGESTITCHER_IMAGESTITCHER_H
#define IMAGESTITCHER_IMAGESTITCHER_H

namespace imageStitcher
{
    struct ImageStitcher
    {
        int keypointsAndDescriptors(std::vector<cv::Mat> &images, std::vector<std::vector<cv::KeyPoint>> &keypoints, std::vector<cv::Mat> &descriptors);
        int matchDescriptors(std::vector<std::vector<cv::DMatch>> &descriptorsMatches, cv::BFMatcher bfMatcher, std::vector<std::vector<cv::KeyPoint>>& keypoints, std::vector<cv::Mat>& descriptors);
        int matchedSrcDst(std::vector<std::vector<cv::Point2f>> &matchedSrc, std::vector<std::vector<cv::Point2f>> &matchedDst, std::vector<std::vector<cv::DMatch>> &descriptorsMatches, std::vector<std::vector<cv::KeyPoint>> &keypoints, std::vector<cv::Mat> &images);
        int combineImages(std::vector<std::vector<cv::DMatch>> &descriptorsMatches, std::vector<std::vector<cv::Point2f>> &matchedSrc, std::vector<std::vector<cv::Point2f>> &matchedDst, std::vector<cv::Mat> &images, cv::Mat &result);
    };
}

#endif //IMAGESTITCHER_IMAGESTITCHER_H

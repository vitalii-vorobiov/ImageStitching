//
// Created by Vitaliy Vorobyov on 25.08.2020.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/features2d/features2d.hpp"

#include "ImageStitcher.h"

int imageStitcher::ImageStitcher::keypointsAndDescriptors(std::vector<cv::Mat>& images, std::vector<std::vector<cv::KeyPoint>>& keypoints, std::vector<cv::Mat>& descriptors)
{
    auto brisk = cv::BRISK::create();
    for (auto & image : images)
    {
        std::vector<cv::KeyPoint> keypoint;
        cv::Mat descriptor;
        brisk->detectAndCompute(image, cv::Mat(), keypoint, descriptor);
        keypoints.push_back(std::move(keypoint));
        descriptors.push_back(std::move(descriptor));
    }
    return 0;
}

int imageStitcher::ImageStitcher::matchDescriptors(std::vector<std::vector<cv::DMatch>> &descriptorsMatches, cv::BFMatcher bfMatcher, std::vector<std::vector<cv::KeyPoint>>& keypoints, std::vector<cv::Mat>& descriptors)
{
    for (int i = 0; i < descriptors.size() - 1; ++i)
    {
        std::vector<cv::DMatch> match;
        bfMatcher.match(descriptors[i], descriptors[i + 1], match);
        descriptorsMatches.push_back(std::move(match));
    }
}

int imageStitcher::ImageStitcher::matchedSrcDst(std::vector<std::vector<cv::Point2f>> &matchedSrc, std::vector<std::vector<cv::Point2f>> &matchedDst, std::vector<std::vector<cv::DMatch>> &descriptorsMatches, std::vector<std::vector<cv::KeyPoint>> &keypoints, std::vector<cv::Mat> &images)
{
    for (int i = 0; i < descriptorsMatches.size(); ++i)
    {
        std::vector<cv::Point2f> src,dst;
        for (int j = 0; j < descriptorsMatches[i].size()/2; j++)
        {
            src.push_back(keypoints[i][descriptorsMatches[i][j].queryIdx].pt+cv::Point2f(0,images[i].rows));
            dst.push_back(keypoints[i+1][descriptorsMatches[i][j].trainIdx].pt);
        }
        matchedSrc.push_back(std::move(src));
        matchedDst.push_back(std::move(dst));
    }
    return 0;
}

int imageStitcher::ImageStitcher::combineImages( std::vector<std::vector<cv::DMatch>> &descriptorsMatches, std::vector<std::vector<cv::Point2f>> &matchedSrc, std::vector<std::vector<cv::Point2f>> &matchedDst, std::vector<cv::Mat> &images, cv::Mat &result)
{
    for (int i = 0; i < descriptorsMatches.size(); ++i)
    {
        auto h = findHomography(matchedSrc[i], matchedDst[i], cv::RANSAC);
//        warpPerspective(images[i+1], result, h.inv(), cv::Size(2*images[i+1].cols +images[i].cols , 2*images[i+1].rows+images[i].rows));
        warpPerspective(images[i+1], result, h.inv(), cv::Size(images[i].cols*3, images[i].rows*3));
//        break;
        std::cout << images[i].rows << std::endl;
        std::cout << images[i].cols << std::endl;
        cv::Mat roi1(result, cv::Rect(0, images[i].rows, images[i].cols, images[i].rows));
        images[i].copyTo(roi1);
    }
    return 0;
}

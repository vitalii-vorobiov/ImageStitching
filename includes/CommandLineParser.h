//
// Created by Vitaliy Vorobyov on 25.08.2020.
//

#ifndef IMAGESTITCHER_COMMANDLINEPARSER_H
#define IMAGESTITCHER_COMMANDLINEPARSER_H

namespace imageStitcher {
    class CommandLineParser {
    private:
        int argc;
        char** argv;
    public:
        CommandLineParser(int argc, char** argv);
        int ParseImages(std::vector<cv::Mat>& images);
    };
}
#endif //IMAGESTITCHER_COMMANDLINEPARSER_H

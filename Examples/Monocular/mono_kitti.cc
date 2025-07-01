/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<vector>
#include<string>
#include<cstdlib>
#include<cstdio>
#include<unistd.h>
#include<sys/stat.h>
#include<opencv2/core/core.hpp>
#include<thread>
#include<numeric>
#include<jsoncpp/json/json.h>
#include<System.h>
//#include"System.h"

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

bool FileExistsAndNotEmpty(const string& filename) {
    struct stat fileStat;
    return (stat(filename.c_str(), &fileStat) == 0 && fileStat.st_size > 0);
}

Json::Value ParseJsonReport(const string& filename) {
    int timeout = 5;
    while (timeout-- > 0) {
        if (FileExistsAndNotEmpty(filename)) {
            break;
        }
        this_thread::sleep_for(chrono::seconds(1));
    }

    ifstream json_file(filename);
    Json::Value root;
    if (json_file.is_open()) {
        json_file >> root;
        json_file.close();
        remove(filename.c_str());
    } else {
        cerr << "Warning: Could not open " << filename << " to read statistics." << endl;
    }
    return root;
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // Launch resource monitor
    pid_t current_pid = getpid();
    char command[256];
    sprintf(command, "python3 resource_monitor.py %d &", current_pid);
    cout << "Launching resource monitor for PID " << current_pid << " with command: " << command << endl;
    int system_ret = system(command);
    if (system_ret != 0) {
        cerr << "Error: Failed to launch resource_monitor.py. Return code: " << system_ret << endl;
    }

    vector<float> vTimesTrack(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = 0;
        if(ni < nImages-1)
            T = vTimestamps[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni-1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Signal monitor to stop
    cout << "Signaling resource monitor to stop..." << endl;
    ofstream stop_file("stop.txt");
    if (stop_file.is_open()) stop_file.close();
    else cerr << "Error: Unable to create stop.txt to signal monitor." << endl;

    SLAM.Shutdown();

    cout << "Waiting for resource monitor to finalize and write report..." << endl;
    this_thread::sleep_for(chrono::seconds(2));

    // Parse report
    Json::Value usage_stats = ParseJsonReport("resource_usage.json");
    remove("stop.txt");

    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    float mean_time = totaltime / nImages;
    float fps = static_cast<float>(nImages) / totaltime;

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << mean_time << endl;

    cout << endl << "======== Overall Evaluation ========" << endl;
    cout << "Total Time = " << totaltime << " s" << endl;
    cout << "Frames processed = " << nImages << endl;
    cout << "FPS = " << fps << " fps" << endl << endl;

    cout << "[AVERAGE USAGE]" << endl;
    Json::StreamWriterBuilder writer;
    writer["indentation"] = " ";
    cout << Json::writeString(writer, usage_stats) << endl;

    SLAM.SaveTrajectoryKITTI("KeyFrameTrajectory.txt");

    return 0;
}\

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}

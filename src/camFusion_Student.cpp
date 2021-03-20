
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same Bounding Box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all LiDAR points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto & lidarPoint : lidarPoints) {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = lidarPoint.x;
        X.at<double>(1, 0) = lidarPoint.y;
        X.at<double>(2, 0) = lidarPoint.z;
        X.at<double>(3, 0) = 1;

        // project LiDAR point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = static_cast<int>(Y.at<double>(0, 0) / Y.at<double>(2, 0)); // pixel coordinates
        pt.y = static_cast<int>(Y.at<double>(1, 0) / Y.at<double>(2, 0));

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current LiDAR point
        for (auto it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = static_cast<int>(it2->roi.x + shrinkFactor * it2->roi.width / 2.0);
            smallerBox.y = static_cast<int>(it2->roi.y + shrinkFactor * it2->roi.height / 2.0);
            smallerBox.width = static_cast<int>(it2->roi.width * (1 - shrinkFactor));
            smallerBox.height = static_cast<int>(it2->roi.height * (1 - shrinkFactor));

            // check weather point is within current bounding box
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check weather point has been enclosed by one or by multiple boxes
        if (1 == enclosingBoxes.size()) {
            // add LiDAR point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(lidarPoint);
        }
    } // eof loop over all LiDAR points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given Bounding Box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Assign every match in the vector kptMatches to the corresponding Bounding Box.
    for(auto& match : kptMatches)
    {
        auto &currKeyPoint = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(currKeyPoint))
        {
            boundingBox.kptMatches.emplace_back(match); // Equivalent to push_back but avoids to have a temp variable.
        }
    }

    double sum = 0;
    //std::cout << "Distance of Matches:" << std::endl;
    // Remove outlier matches based on the euclidean distance between them in relation to all the matches in the Bounding Box.
    for (auto& it : boundingBox.kptMatches)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it.trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it.queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt); // Calculats the absolute distante between the 2 kpts.
        sum += dist;    // Total distance between all keypoints.
        // Print all distances, not needed in final version, just for information.
        //std::cout << dist << "; ";
    }
    //std::cout << std::endl;
    // Compute the mean of the distance between matches inside the Bounding Box.
    double mean = sum / boundingBox.kptMatches.size();

    double ratio = 1.5;
    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();)
    {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
        double dist = cv::norm(kpCurr.pt - kpPrev.pt);

        if (dist >= mean * ratio)
        {
            // If distance between botch kpts matches is bigger than the stablished threshold, it is deleted from vector.
            boundingBox.kptMatches.erase(it);
        }
        else
        {
            // If distance between botch kpts matches is under the threshold, it is kept. We go to evaluate next pair.
            it++;
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // Vector to store the distance ratios for all keypoints between current and previous frames.
    vector<double> distRatios; 
    // Compute distance ratios between all matched keypoints.
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++)
    {
        // Get current keypoint and its matched partner in the previous frame.
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++)
        {
            // Minimum required distance.
            double minDist = 100.0;

            // Get next keypoint and its matched partner in the previous frame.
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // Compute distances and distance ratios.
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            // Avoid division by zero in order to not have crashes with an operation.
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // EOF: inner loop over all matched kpts.
    }     // EOF: outer loop over all matched kpts.

    // If list of distance ratios is empty, there is no TTC to be computed, so we can exit the function
    // with value NAN.
    if (distRatios.empty())
    {
        TTC = NAN;
        return;
    }
    // Get all distance ratios sorted.
    std::sort(distRatios.begin(), distRatios.end());
    // Print all the distance ratios. Only for infomation purposes, so commented from final version.
    /*std::cout << "Distance Ratios:" << std::endl;
    for (const auto& dist : distRatios) {
        std::cout << dist << "; ";
    }
    std::cout << std::endl;*/


    long medIndex = floor(distRatios.size() / 2.0);
    // Compute the median distance ratio to remove any outlier influence.
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    std::cout << "medDistRatio = " << medDistRatio << std::endl;
    // Compute the TTC.
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    std::cout << "TTC camera = " << TTC << std::endl;
}


/*
* Support function for euclideanCluster.
*/
void proximity(int index, std::vector<std::vector<float>> points, std::vector<int> &cluster, KdTree *tree,  float distanceTol, std::vector<bool> &processedPoints)
{
		processedPoints[index]= true;
		// Store index of the processed point
		cluster.push_back(index);
		std::vector<int> near_points= tree->search(points[index],distanceTol);
		for (auto index: near_points)
		{
			// If point has not been really processed, we ensure it is processed
			if(!processedPoints[index])
				proximity(index, points, cluster, tree, distanceTol, processedPoints);
		}
}

/*
* The function ClusterHelper is used to get all the clusters of lidar points in the Bounding Box
* in order to filter out the outliers. The euclidean distance is used to determine the clusters.
*/
std::vector<std::vector<int>> euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol)
{
	std::vector<std::vector<int>> clusters;
	// Vector to allocate a single cluster
	std::vector<bool> processedPoints(points.size());
	std::fill (processedPoints.begin(),processedPoints.end(),false);
	for(int i=0;i<points.size();i++)
    {
		// If a point is already processed, nothing to be done for it
		if(processedPoints[i])
			continue;
		// Vector to allocate the indices of processed points
		std::vector <int> cluster;
		proximity(i, points, cluster, tree,  distanceTol, processedPoints);
		// Save the vector with the points which are already processed
		clusters.push_back(cluster);
	}

 
	return clusters;

}


std::vector<LidarPoint> removeLidarOutlier(const std::vector<LidarPoint> &lidarPoints, float clusterTolerance) {
    //auto treePrev = std::make_shared<KdTree>();
    KdTree* treePrev = new KdTree;
    std::vector<std::vector<float>> points;
    for (int i=0; i< lidarPoints.size(); i++)
    {
        std::vector<float> point({static_cast<float>(lidarPoints[i].x),
                                  static_cast<float>(lidarPoints[i].y),
                                  static_cast<float>(lidarPoints[i].z)});
        points.push_back(point);
        treePrev->insert(points[i], i);
    }
    // Apply the euclidean distance clustering to the data. 
    std::vector<std::vector<int>> cluster_indices = euclideanCluster(points, treePrev, clusterTolerance);
    // Determine the different clusters that are present and their sizes.
    std::vector<LidarPoint> maxLidarPointsCluster;
    for (const auto& get_indices : cluster_indices)
    {
        std::vector<LidarPoint> temp;
        for (const auto index : get_indices)
        {
            temp.push_back(lidarPoints[index]);
        }

        std::cout << "Cluster size = " << temp.size() << std::endl;
        // Determine if the actual one is the biggest cluster, which represents the main body we are following. 
        if (temp.size() > maxLidarPointsCluster.size())
        {
            maxLidarPointsCluster = std::move(temp);
        }
    }

    std::cout << "Max cluster size = " << maxLidarPointsCluster.size() << std::endl;
    return maxLidarPointsCluster;
}


void computeTTCLidar(const std::vector<LidarPoint> &lidarPointsPrev,
                     const std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    // Auxiliary variables.
    double dT = 1 / frameRate;              // Time between two measurements in seconds.
    constexpr double laneWidth = 4.0;       // Assumed width of the ego lane.
    constexpr float clusterTolerance = 0.1; // Tolerance for the euclidean clustering.

    // Find closest distance to LiDAR points within ego lane.
    // Initial value is big to avoid having issues of initialization and conflicts with any possible
    // minimal distance that could be big.
    double minXPrev = 1e9, minXCurr = 1e9;

    // Remove outliers by using this assisting function, made ad-hoc for this purpose.
    // It is used for data of both previous and current frames.
    std::cout << "Process previous frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsPrevClustered = removeLidarOutlier(lidarPointsPrev, clusterTolerance);
    std::cout << "Process current frame..." << std::endl;
    std::vector<LidarPoint> lidarPointsCurrClustered = removeLidarOutlier(lidarPointsCurr, clusterTolerance);

    // Check the 3D Lidar points that are within the lane of the ego vehicle, previous frame.
    for (const auto & it : lidarPointsPrevClustered)
    {
        if (abs(it.y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXPrev = it.x < minXPrev ? it.x : minXPrev;
        }
    }
    // Check the 3D Lidar points that are within the lane of the ego vehicle, current frame.
    for (const auto & it : lidarPointsCurrClustered)
    {
        if (abs(it.y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            minXCurr = it.x < minXCurr ? it.x : minXCurr;
        }
    }

    std::cout << "Prev min X = " << minXPrev << std::endl;
    std::cout << "Curr min X = " << minXCurr << std::endl;

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
    std::cout << "TTC Lidar = " << TTC << std::endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    int previous = prevFrame.boundingBoxes.size();  // Bounding Boxes in the previous frame.
    int current = currFrame.boundingBoxes.size();   // Bounding Boxes in the current frame.
    // 2D array to store which keypoints are inside the Bounding Boxes in both the previous and
    // the current frames. 1 for indicating its presence and 0 for the opposite.
    int pt_counts[previous][current] = { };

    // Iteration through all the matches.
    for (auto it = matches.begin(); it != matches.end() - 1; ++it)     
    {
        // Following code is done after consulting official documentation in 
        // https://docs.opencv.org/4.1.0/db/d39/classcv_1_1DescriptorMatcher.html#a0f046f47b68ec7074391e1e85c750cba
        // prevFrame.keypoints is indexed by queryIdx.
        // currFrame.keypoints is indexed by trainIdx.

        // Keypoint in the Previous Frame.
        cv::KeyPoint query_kpt = prevFrame.keypoints[it->queryIdx];
        auto query_pt = cv::Point(query_kpt.pt.x, query_kpt.pt.y);  // Take the coordinates of the keypoint.
        bool query_found = false;   // Initially, I suppose it is not found.

        // Keypoint in the Current Frame.
        cv::KeyPoint train_kpt = currFrame.keypoints[it->trainIdx];
        auto train_pt = cv::Point(train_kpt.pt.x, train_kpt.pt.y);  // Take the coordinates of the keypoint.
        bool train_found = false;   // Initially, I suppose it is not found.

        // Vectors to store the ids of the keypoints that are found in the Bounding Boxes
        // for both the previous and the current frames.
        std::vector<int> query_id, train_id;
        int temp_query_id, temp_train_id;
        // Check the previous frame and store the ids of all keypoints found in the Bounding Boxes. 
        for (int i = 0; i < previous; i++) 
        {
            if (prevFrame.boundingBoxes[i].roi.contains(query_pt))            
             {
                query_found = true;
                temp_query_id = i;
                //query_id.push_back(i);  // If found in the Bounding Boxes of the previous frame, it is stored.
             }
        }
        // Check the current frame and store the ids of all keypoints found in the Bounding Boxes.
        for (int i = 0; i < current; i++) 
        {
            if (currFrame.boundingBoxes[i].roi.contains(train_pt))            
            {
                train_found = true;
                temp_train_id = i;
                //train_id.push_back(i);  // If found in the Bounding Boxes of the current frame, it is stored.
            }
        }

        // Only if the keypoint is found in both the previous and the current frames
        // we increment the counting of the points detected in both frames.
        if (query_found && train_found) 
        {
            // Only add the last pair keypoints (corresponding to the match we are currently analyzing)
            // in case both of them are inside Bounding Boxes
            // If only one is added to the corresponding vector, but not the other, we are going to have
            // vectors of different length that will not correspond to the number of matches.
            //query_id.push_back(temp_query_id);  // If found in the Bounding Boxes of the previous frame, it is stored.
            //train_id.push_back(temp_train_id);  // If found in the Bounding Boxes of the current frame, it is stored.
            
            //This double loop will go through all the stored ids of boundingBoxes
            /*for (auto id_prev: query_id)
                for (auto id_curr: train_id)
                    pt_counts[id_prev][id_curr] += 1;*/
            // Optimization of the loop : just the last added ids.
            
            //auto id_prev = *(query_id.end()-1);
            //auto id_curr = *(train_id.end()-1);
            auto id_prev = temp_query_id;
            auto id_curr = temp_train_id;
            // +1 to the combination of boundig boxes of previous and current frame
            // with these specific indexes 
            pt_counts[id_prev][id_curr] += 1;       
        }
    }

    // Double loop to go through all the Bounding Boxes in both frames.
    for (int i = 0; i < previous; i++)  // Iterate through the previous frame.
    {  
        int max_count = 0;
        int id_max = 0;
        
        for (int ii = 0; ii < current; ii++)    // Iterate through the current frame.
        {
            if (pt_counts[i][ii] > max_count)
            {  
                max_count = pt_counts[i][ii];
                id_max = ii;
            }
        }
        // Take the id of the match found as best result for the selected keypoint of previous frame in the
        // actual one. 
        bbBestMatches[i] = id_max;

        //cout << "BoundingBox " << i << " cooresponds to " << id_max << std::endl;
    } 
}

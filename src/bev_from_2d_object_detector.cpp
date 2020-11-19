/*
Created on Thu Aug  6 11:27:43 2020

@author: Carlos Gómez-Huélamo, Rodrigo Gutiérrez Moreno and Javier Araluce Ruiz

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: 
Outputs: 

Note that 

Executed via 
*/

// General purpose includes

#include <math.h>

// ROS includes

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Custom includes

#include "darknet_ros_msgs/BoundingBox.h"
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "t4ac_msgs/BEV_detection.h"
#include "t4ac_msgs/BEV_detections_list.h"

// OpenCV includes

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Point
{
    public:
        Point(int x, int y)
        {
            this->x = x;
            this->y = y;
        }

        int get_x()
        {
            return x;
        }

        int get_y()
        {
            return y;
        }
    private:
        int x, y;
};

class Node
{
    private:
        ros::NodeHandle nh;

        ros::Publisher pub_detected_obstacles;
        ros::Publisher pub_detected_road_signals;
        ros::Publisher pub_detected_bev_image_detections_marker_array;

        message_filters::Subscriber<sensor_msgs::Image> image_left_sub;
        message_filters::Subscriber<sensor_msgs::Image> image_right_sub;
        message_filters::Subscriber<sensor_msgs::Image> elas_sub;
        message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> darknet_detections_sub;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
        typedef message_filters::Synchronizer<MySyncPolicy> Sync;
        boost::shared_ptr<Sync> sync_;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> MySyncPolicy2;
        typedef message_filters::Synchronizer<MySyncPolicy2> Sync2;
        boost::shared_ptr<Sync2> sync2_;

        cv::Ptr<cv::StereoSGBM> sgbm;
    public:
        // Constructor

        Node(cv::Ptr<cv::StereoSGBM> sgbm, bool use_elas)
        {
            pub_detected_obstacles = nh.advertise<t4ac_msgs::BEV_detections_list>("/perception/detection/bev_image_obstacles", 5, true);
            pub_detected_road_signals = nh.advertise<t4ac_msgs::BEV_detections_list>("/perception/detection/bev_image_road_signals", 5, true);
            pub_detected_bev_image_detections_marker_array = nh.advertise<visualization_msgs::MarkerArray>("/perception/detection/bev_image_detections_marker", 5, true);

            image_left_sub.subscribe(nh, "/stereo_center/left/image", 10);
            image_right_sub.subscribe(nh, "/stereo_center/right/image", 10);
            elas_sub.subscribe(nh, "/elas/depth", 10);
            darknet_detections_sub.subscribe(nh, "/darknet_ros/bounding_boxes", 10);

            if (use_elas)
            {
                sync_.reset(new Sync(MySyncPolicy(100), elas_sub, darknet_detections_sub));
                sync_->registerCallback(boost::bind(&Node::bev_from_2d_object_detector_callback, this, _1, _2));
            }
            else
            {
                sync2_.reset(new Sync2(MySyncPolicy2(10), image_left_sub, image_right_sub, darknet_detections_sub));
                sync2_->registerCallback(boost::bind(&Node::bev_from_2d_object_detector_callback2, this, _1, _2, _3));
            }

            // Initialize private atributes 

            this->sgbm = sgbm; 
        }

        double compute_median(cv::Mat bounding_box)
        {
            bounding_box = bounding_box.reshape(0,1); // Spread input bounding box to a single row
            std::vector<double> bounding_box_vector;
            bounding_box.copyTo(bounding_box_vector);
            std::nth_element(bounding_box_vector.begin(), bounding_box_vector.begin() + bounding_box_vector.size()/2, bounding_box_vector.end());

            return bounding_box_vector[bounding_box_vector.size()/2];
        }

        void disp2depth_bbox(cv::Mat disparity, double baseline, double f, Point bounding_box_centroid, 
                             Point image_center, t4ac_msgs::BEV_detection& bev_image_detection)
        {
            double median = compute_median(disparity);
            median = median / 10; // ?
            double z;

            if (!isnan(median))
            {
                z = f * (baseline / median);
                bev_image_detection.x = z; 
                bev_image_detection.y = -(z * (bounding_box_centroid.get_x()-image_center.get_x())) / f;  
            }
            else
            {
                bev_image_detection.x = 50000;
                bev_image_detection.y = 50000;
            }
        }

        void depth_bbox(cv::Mat depth_bb, double f, Point bounding_box_centroid, 
                        Point image_center, t4ac_msgs::BEV_detection& bev_image_detection)
        {
            double z = compute_median(depth_bb);

            if (!isnan(z))
            {
                bev_image_detection.x = z - 0.7; // 0.7 is the x-distance between the left camera and lidar (lidar frame)
                bev_image_detection.y = (-(z * (bounding_box_centroid.get_x()-image_center.get_x())) / f) - 0.06; // 0.06 is the y-distance between the left camera and lidar (lidar frame)   
            }
            else
            {
                bev_image_detection.x = 50000;
                bev_image_detection.y = 50000;
            }
        }

        void bev_from_2d_object_detector_callback(const sensor_msgs::Image::ConstPtr& depth_msg, 
                                                  const darknet_ros_msgs::BoundingBoxes::ConstPtr& darknet_detections_msg)
        {
            Point image_center(depth_msg->width/2, depth_msg->height/2);
            double f = 935.3074;

            t4ac_msgs::BEV_detections_list bev_image_obstacles, bev_image_road_obstacles;
            visualization_msgs::MarkerArray bev_image_detections_marker_array;

            bev_image_obstacles.header.stamp = bev_image_road_obstacles.header.stamp = depth_msg->header.stamp;

            for (size_t i = 0; i < darknet_detections_msg->bounding_boxes.size(); i++)
            {
                int xmin, ymin, xmax, ymax;
                cv::Mat depth;
                cv_bridge::CvImageConstPtr depth_img_cv;
                try
                {
                   depth_img_cv = cv_bridge::toCvShare (depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
                   depth_img_cv->image.convertTo(depth, CV_32F, 0.001);
                 }
                 catch (cv_bridge::Exception& e)
                 {
                   ROS_ERROR("cv_bridge exception: %s", e.what());
                  return;
                 }

                xmin = darknet_detections_msg->bounding_boxes[i].xmin;
                ymin = darknet_detections_msg->bounding_boxes[i].ymin;
                xmax = darknet_detections_msg->bounding_boxes[i].xmax;
                ymax = darknet_detections_msg->bounding_boxes[i].ymax;

                std::string bb_type = darknet_detections_msg->bounding_boxes[i].Class;
                std::string traffic_string = "traffic"; // Traffic signal and traffic light
                std::string stop_string = "stop"; // Stop signal (vertical and road)
                int pos = 0;
                
                cv::Mat aux(depth, cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                cv::Mat depth_bb;
                aux.copyTo(depth_bb);
                Point bounding_box_centroid((xmin+xmax)/2, (ymin+ymax)/2);

                t4ac_msgs::BEV_detection bev_image_detection;

                bev_image_detection.type = darknet_detections_msg->bounding_boxes[i].Class;
                bev_image_detection.score = darknet_detections_msg->bounding_boxes[i].probability;

                depth_bbox(depth_bb, f, bounding_box_centroid, image_center, bev_image_detection);

                if ((bb_type.find(traffic_string,pos) != std::string::npos) || (bb_type.find(stop_string,pos) != std::string::npos))
                {
                    bev_image_road_obstacles.bev_detections_list.push_back(bev_image_detection);
                } 
                else
                {
                    bev_image_obstacles.bev_detections_list.push_back(bev_image_detection);
                }

                visualization_msgs::Marker bev_image_detection_marker;

                bev_image_detection_marker.header.frame_id = "velodyne";
                bev_image_detection_marker.ns = "bev_image_detections";
				bev_image_detection_marker.id = i;
				bev_image_detection_marker.action = visualization_msgs::Marker::ADD;
				bev_image_detection_marker.type = visualization_msgs::Marker::CUBE;
				bev_image_detection_marker.lifetime = ros::Duration(0.30);
				bev_image_detection_marker.pose.position.x = bev_image_detection.x;
				bev_image_detection_marker.pose.position.y = bev_image_detection.y;
				bev_image_detection_marker.pose.position.z = -1.5;
				bev_image_detection_marker.scale.x = 1;
				bev_image_detection_marker.scale.y = 1;
				bev_image_detection_marker.scale.z = 1;
				bev_image_detection_marker.color.r = 255;
				bev_image_detection_marker.color.g = 0;
				bev_image_detection_marker.color.b = 0;
				bev_image_detection_marker.color.a = 0.8;

                bev_image_detections_marker_array.markers.push_back(bev_image_detection_marker);
            }

            pub_detected_road_signals.publish(bev_image_road_obstacles);
            pub_detected_obstacles.publish(bev_image_obstacles);  
            pub_detected_bev_image_detections_marker_array.publish(bev_image_detections_marker_array);
        }

        void bev_from_2d_object_detector_callback2(const sensor_msgs::ImageConstPtr& img_left_msg, 
                                                   const sensor_msgs::ImageConstPtr& img_right_msg, 
                                                   const darknet_ros_msgs::BoundingBoxes::ConstPtr& darknet_detections_msg)
        {
            cv::Mat cv_img_left, cv_img_left_gray;
            cv::Mat cv_img_right, cv_img_right_gray;
            cv::Mat disp, disp8, disp_bb;

            // Camera parameters

            double baseline = 0.12; // m
            int image_width = 1080; // Pixels
            int fov = 70; // Degrees
            double f = image_width / (2.0 * tan(fov * M_PI / 360.0)); // Focal length (mm)

            try
            {
                cv_img_left = cv_bridge::toCvShare(img_left_msg, "bgra8")->image;
                cv_img_right = cv_bridge::toCvShare(img_right_msg, "bgra8")->image;
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            Point image_center(cv_img_left.cols/2, cv_img_left.rows/2);

            cvtColor(cv_img_left, cv_img_left, CV_BGR2GRAY);
            cvtColor(cv_img_right, cv_img_right_gray, CV_BGR2GRAY);
            Node::sgbm->compute(cv_img_left, cv_img_right_gray, disp);

            //cv::normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U); // To display

            t4ac_msgs::BEV_detections_list bev_image_obstacles, bev_image_road_obstacles;

            bev_image_obstacles.header.stamp = bev_image_road_obstacles.header.stamp = img_left_msg->header.stamp;
            bev_image_obstacles.header.frame_id = bev_image_road_obstacles.header.frame_id = img_left_msg->header.frame_id;

            for (size_t i = 0; i < darknet_detections_msg->bounding_boxes.size(); i++)
            {
                int xmin, ymin, xmax, ymax;

                xmin = darknet_detections_msg->bounding_boxes[i].xmin;
                ymin = darknet_detections_msg->bounding_boxes[i].ymin;
                xmax = darknet_detections_msg->bounding_boxes[i].xmax;
                ymax = darknet_detections_msg->bounding_boxes[i].ymax;

                std::string bb_type = darknet_detections_msg->bounding_boxes[i].Class;
                std::string traffic_string = "traffic"; // Traffic signal and traffic light
                std::string stop_string = "stop"; // Stop signal (vertical and road)
                int pos = 0;

                cv::Mat aux(disp, cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
                aux.copyTo(disp_bb);
                Point bounding_box_centroid((xmin+xmax)/2, (ymin+ymax)/2);

                t4ac_msgs::BEV_detection bev_image_detection;

                bev_image_detection.type = darknet_detections_msg->bounding_boxes[i].Class;
                bev_image_detection.score = darknet_detections_msg->bounding_boxes[i].probability;

                disp2depth_bbox(disp_bb, baseline, f, bounding_box_centroid, image_center, bev_image_detection);

                if ((bb_type.find(traffic_string,pos) != std::string::npos) && (bb_type.find(stop_string,pos) != std::string::npos))
                {
                    bev_image_road_obstacles.bev_detections_list.push_back(bev_image_detection);
                }
                else
                {
                    bev_image_obstacles.bev_detections_list.push_back(bev_image_detection);
                }
            }

            pub_detected_road_signals.publish(bev_image_road_obstacles);
            pub_detected_obstacles.publish(bev_image_obstacles); 
        }
};

int main(int argc, char **argv)
{
    // Init ROS node

    ros::init(argc, argv, "bev_from_2d_object_detector_node"); 

    // Specify parameters to create the depth map

    int minDisparity = 0;
    int numDisparities = 16 * 3;
    int blockSize = 3;
    int P1 = 8*3*blockSize;
    int P2 = 32*3*blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 0;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 1;
    int mode = cv::StereoSGBM::MODE_SGBM;  

    bool use_elas = true; 

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap,
                                                          uniquenessRatio, speckleWindowSize, speckleRange, mode);

    Node synchronizer(sgbm, use_elas);

    // ROS spin

    ros::spin();
}
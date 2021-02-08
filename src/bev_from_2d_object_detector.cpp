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
#include <Point.hpp>

// OpenCV includes

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Node
{
    private:
        double fov;

        ros::NodeHandle nh;

        ros::Publisher pub_detected_obstacles;
        ros::Publisher pub_detected_road_signals;
        ros::Publisher pub_detected_bev_image_detections_marker_array;

        message_filters::Subscriber<sensor_msgs::Image> depth_map_sub;
        message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes> darknet_detections_sub;

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
        typedef message_filters::Synchronizer<MySyncPolicy> Sync;
        boost::shared_ptr<Sync> sync_;

    public:
        // Constructor

        Node(double fov)
        {
            pub_detected_obstacles = nh.advertise<t4ac_msgs::BEV_detections_list>("/t4ac/perception/detection/bev_image_obstacles", 5, true);
            pub_detected_road_signals = nh.advertise<t4ac_msgs::BEV_detections_list>("/t4ac/perception/detection/bev_image_road_signals", 5, true);
            pub_detected_bev_image_detections_marker_array = nh.advertise<visualization_msgs::MarkerArray>("/t4ac/perception/detection/bev_image_detections_marker", 5, true);

            depth_map_sub.subscribe(nh, "/carla/ego_vehicle/camera/depth/front/image_depth", 1000);
            darknet_detections_sub.subscribe(nh, "/darknet_ros/bounding_boxes", 1000);

            sync_.reset(new Sync(MySyncPolicy(100), depth_map_sub, darknet_detections_sub));
            sync_->registerCallback(boost::bind(&Node::bev_from_2d_object_detector_callback, this, _1, _2));

            // Specify FoV

            this->fov = fov;
        }

        double compute_median(cv::Mat bounding_box)
        {
            bounding_box = bounding_box.reshape(0,1); // Spread input bounding box to a single row
            std::vector<double> bounding_box_vector;
            bounding_box.copyTo(bounding_box_vector);
            std::nth_element(bounding_box_vector.begin(), bounding_box_vector.begin() + bounding_box_vector.size()/2, bounding_box_vector.end());

            return bounding_box_vector[bounding_box_vector.size()/2];
        }

        void depth_bbox(cv::Mat depth_bb, double f, Point bounding_box_centroid, 
                        Point image_center, t4ac_msgs::BEV_detection& bev_image_detection)
        {
            double z = compute_median(depth_bb);

            if (!isnan(z))
            {
                bev_image_detection.x = z - 0.41; // 0.41 is the x-distance between camera and lidar (lidar frame)
                bev_image_detection.y = (-(z * (bounding_box_centroid.get_x()-image_center.get_x())) / f); // 0.06 is the y-distance between the left camera and lidar (lidar frame)   
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
            cv::Mat image_depth;

            Point image_center(depth_msg->width/2, depth_msg->height/2);

            double f = depth_msg->width / (2 * tan(fov * M_PI / 360));    

            t4ac_msgs::BEV_detections_list bev_image_obstacles, bev_image_road_obstacles;
            visualization_msgs::MarkerArray bev_image_detections_marker_array;

            bev_image_obstacles.header.stamp = bev_image_road_obstacles.header.stamp = depth_msg->header.stamp;

            for (size_t i = 0; i < darknet_detections_msg->bounding_boxes.size(); i++)
            {
                
                
                if (darknet_detections_msg->bounding_boxes[i].probability > 0.0)
                {
                    int xmin, ymin, xmax, ymax;

                    try
                {
                    image_depth = cv_bridge::toCvShare(depth_msg)->image;
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
                
                cv::Mat aux(image_depth, cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin));
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

                bev_image_detection_marker.header.frame_id = "ego_vehicle/lidar/lidar1";
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
            }

            pub_detected_road_signals.publish(bev_image_road_obstacles);
            pub_detected_obstacles.publish(bev_image_obstacles);  
            pub_detected_bev_image_detections_marker_array.publish(bev_image_detections_marker_array);
        }
};

int main(int argc, char **argv)
{
    // Init ROS node

    ros::init(argc, argv, "bev_from_2d_object_detector_node"); 

    double fov = 85;

    Node synchronizer(fov);

    // ROS spin

    ros::spin();
}
/*
Created on Thu Aug  6 11:27:43 2020

@author: Carlos Gómez-Huélamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: 
Outputs: 

Note that 

Executed via 
*/

// Includes //

// General purpose includes

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include <string.h>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <vector>

// Custom includes

#include <Hungarian.hpp>
#include <Point.hpp>

// ROS includes

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// Techs4AgeCar includes

#include "t4ac_msgs/BEV_detections_list.h"
#include "t4ac_msgs/BEV_trackers_list.h"

// End Includes //


// Global variables //

double road_curvature = 0;
double max_road_curvature = 20;

// End Global variables //


// ROS communications //

// Publishers

ros::Publisher pub_merged_obstacles_marker_array;
ros::Publisher pub_merged_obstacles;
ros::Publisher pub_monitorized_area;

// Subcribers

ros::Subscriber sub_road_curvature;

// End ROS communications //


// Declarations of functions //

double euclidean_distance(Point , Point );
bool inside_monitorized_area(Point , float []);

// ROS Callbacks

void sensor_fusion_callback(const t4ac_msgs::BEV_detections_list::ConstPtr& , 
							const t4ac_msgs::BEV_detections_list::ConstPtr& );
void road_curvature_cb(const std_msgs::Float64::ConstPtr& );

// End Declarations of functions //


// Main //

int main (int argc, char ** argv)
{
	// Initialize ROS

	ros::init(argc, argv, "sensor_fusion_node");
	ros::NodeHandle nh;

    // Publishers

    pub_merged_obstacles_marker_array = nh.advertise<visualization_msgs::MarkerArray>("/perception/detection/merged_obstacles_marker", 100, true);
	pub_merged_obstacles = nh.advertise<t4ac_msgs::BEV_detections_list>("/perception/detection/merged_obstacles", 100, true);
	pub_monitorized_area = nh.advertise<visualization_msgs::Marker>("/perception/detection/monitorized_area_marker", 100, true);

	// Subscribers

	sub_road_curvature = nh.subscribe("/control/rc", 10, road_curvature_cb);

	message_filters::Subscriber<t4ac_msgs::BEV_detections_list> sub_bev_image_detections;
	message_filters::Subscriber<t4ac_msgs::BEV_detections_list> sub_bev_lidar_detections;

	sub_bev_image_detections.subscribe(nh, "/perception/detection/bev_image_obstacles", 10);
	sub_bev_lidar_detections.subscribe(nh, "/perception/detection/bev_lidar_obstacles", 10);

	// Callback 1: Synchronize LiDAR point cloud based BEV detections and Depth map & 2D Object detectio based BEV detections

	typedef message_filters::sync_policies::ApproximateTime<t4ac_msgs::BEV_detections_list, t4ac_msgs::BEV_detections_list> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), sub_bev_image_detections, sub_bev_lidar_detections);
	sync.registerCallback(boost::bind(&sensor_fusion_callback, _1, _2));

	ros::spin ();
}

// End Main //


// Definitions of functions //

double euclidean_distance(Point bev_image_detection, Point bev_lidar_detection)
{
	double x_diff = bev_image_detection.get_x()-bev_lidar_detection.get_x();
	double y_diff = bev_image_detection.get_y()-bev_lidar_detection.get_y();
	double ed = double(sqrt(pow(x_diff,2)+pow(y_diff,2)));
	return ed;
}

bool inside_monitorized_area(Point bev_detection, float monitorized_area[])
{
	if (bev_detection.get_x() < monitorized_area[0] && bev_detection.get_x() > monitorized_area[1]
		&& bev_detection.get_y() < monitorized_area[2] && bev_detection.get_y() > monitorized_area[3])
		return true;
	else
		return false;
}

// Callbacks

void road_curvature_cb(const std_msgs::Float64::ConstPtr& road_curvature_msg)
{
	road_curvature = road_curvature_msg->data;

	visualization_msgs::Marker monitorized_area_marker;

	float xmax, xmin, ymax, ymin, a;
	
	a = road_curvature/max_road_curvature;
	xmax = a*14.0;
    xmin = 0;
    ymax = a*2.0;
    ymin = a*(-2.5);

	monitorized_area_marker.header.frame_id = "velodyne";
	monitorized_area_marker.type = visualization_msgs::Marker::CUBE;
	monitorized_area_marker.lifetime = ros::Duration(0.40);
	monitorized_area_marker.pose.position.x = xmax/2;
	monitorized_area_marker.pose.position.y = (ymin+ymax)/2;
	monitorized_area_marker.pose.position.z = -2;
	monitorized_area_marker.scale.x = xmax;
	monitorized_area_marker.scale.y = abs(ymin)+ymax;
	monitorized_area_marker.scale.z = 0.3;
	monitorized_area_marker.color.r = 255;
	monitorized_area_marker.color.g = 255;
	monitorized_area_marker.color.b = 255;
	monitorized_area_marker.color.a = 0.4;

	pub_monitorized_area.publish(monitorized_area_marker);
}

void sensor_fusion_callback(const t4ac_msgs::BEV_detections_list::ConstPtr& bev_image_detections_msg, 
                            const t4ac_msgs::BEV_detections_list::ConstPtr& bev_lidar_detections_msg)
{
	visualization_msgs::MarkerArray merged_obstacles_marker_array;
	t4ac_msgs::BEV_detections_list merged_obstacles;
	std::vector<int> evaluated_lidar_obstacles;

	// Define monitorized area

	float xmax, xmin, ymax, ymin, a;
	
	a = road_curvature/max_road_curvature;
	xmax = a*14.0;
    xmin = 0;
    ymax = a*2.0;
    ymin = a*(-2.5);

	float monitorized_area[] = {xmax,xmin,ymax,ymin};

	// Associate bev_detections using GNN (Global Nearest Neighbour) algorithm. TODO: Improve this late fusion

	for (size_t i=0; i<bev_image_detections_msg->bev_detections_list.size(); i++)
	{
		float max_diff = 4; // Initialize maximum allowed difference
		int index_most_similar = -1;

		float x_image, y_image; // LiDAR frame (BEV information based on the depth map)
		x_image = bev_image_detections_msg->bev_detections_list[i].x;
		y_image = bev_image_detections_msg->bev_detections_list[i].y;

		string type = bev_image_detections_msg->bev_detections_list[i].type;
		float score = bev_image_detections_msg->bev_detections_list[i].score;

		Point bev_image_detection(x_image, y_image);

		if (bev_lidar_detections_msg->bev_detections_list.size() > 0)
		{
			float x_closest_lidar, y_closest_lidar;
			x_closest_lidar = y_closest_lidar = 0;

			for (size_t j=0; j<bev_lidar_detections_msg->bev_detections_list.size(); j++)
			{
				float x_aux = (bev_lidar_detections_msg->bev_detections_list[j].x_corners[2]+
				               bev_lidar_detections_msg->bev_detections_list[j].x_corners[3]) / 2;
				x_aux += bev_lidar_detections_msg->bev_detections_list[j].x;

				float y_aux = (bev_lidar_detections_msg->bev_detections_list[j].y_corners[2]+
				               bev_lidar_detections_msg->bev_detections_list[j].y_corners[3]) / 2;
				y_aux += bev_lidar_detections_msg->bev_detections_list[j].y;
				
				Point bev_lidar_detection(x_aux, y_aux);

				if (inside_monitorized_area(bev_lidar_detection,monitorized_area))
				{
					double ed = euclidean_distance(bev_image_detection, bev_lidar_detection);

					if (ed < max_diff)
					{
						max_diff = ed;
						index_most_similar = j;
						x_closest_lidar = x_aux;
						y_closest_lidar = y_aux;
					}
				}
			}

			if ((index_most_similar != -1 && 
			    !(std::find(evaluated_lidar_obstacles.begin(), evaluated_lidar_obstacles.end(), index_most_similar) != evaluated_lidar_obstacles.end()))
				|| inside_monitorized_area(bev_image_detection,monitorized_area))
			{
				evaluated_lidar_obstacles.push_back(index_most_similar);

				// Visual marker of merged obstacle

				visualization_msgs::Marker merged_obstacle_marker;

				merged_obstacle_marker.header.frame_id = bev_lidar_detections_msg->header.frame_id;
				merged_obstacle_marker.ns = "merged_obstacles";
				merged_obstacle_marker.id = index_most_similar;
				merged_obstacle_marker.action = visualization_msgs::Marker::ADD;
				merged_obstacle_marker.type = visualization_msgs::Marker::CUBE;
				merged_obstacle_marker.lifetime = ros::Duration(0.40);

				if (x_closest_lidar != 0 && y_closest_lidar != 0)
				{
					merged_obstacle_marker.pose.position.x = x_closest_lidar;
					merged_obstacle_marker.pose.position.y = y_closest_lidar;
				}
				else
				{
					merged_obstacle_marker.pose.position.x = x_image;
					merged_obstacle_marker.pose.position.y = y_image;
				}
				
				merged_obstacle_marker.pose.position.z = 0.5;
				merged_obstacle_marker.scale.x = 1;
				merged_obstacle_marker.scale.y = 1;
				merged_obstacle_marker.scale.z = 1;
				merged_obstacle_marker.color.r = 0;
				merged_obstacle_marker.color.g = 255;
				merged_obstacle_marker.color.b = 0;
				merged_obstacle_marker.color.a = 0.8;

				// T4AC BEV detection of merged obstacle -> Used by decision-making layer

				t4ac_msgs::BEV_detection merged_obstacle;

				merged_obstacle.type = type;
                merged_obstacle.score = score;

				if (x_closest_lidar != 0 && y_closest_lidar != 0)
				{
					merged_obstacle.x = x_closest_lidar;
					merged_obstacle.y = y_closest_lidar;
				}
				else
				{
					merged_obstacle.x = x_image;
					merged_obstacle.y = y_image;
				}

				merged_obstacles_marker_array.markers.push_back(merged_obstacle_marker);
				merged_obstacles.bev_detections_list.push_back(merged_obstacle);
			}
		}
	}

	// Publish merged obstacles

	pub_merged_obstacles_marker_array.publish(merged_obstacles_marker_array);
	pub_merged_obstacles.publish(merged_obstacles);
}

// End Definitions of functions //

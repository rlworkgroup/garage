//
// Created by hejia on 6/28/18.
//

#include <ros/ros.h>
#include <intera_core_msgs/EndpointState.h>
#include <std_msgs/Bool.h>
#include <fcl.h>
#include "sawyer_gripper.h"

int main(int argc, char **argv) {
    using CollisionGeometryPtr_t = std::shared_ptr<fcl::CollisionGeometryd>;

    ros::init(argc, argv, "collision_avoidance_node");

    ros::NodeHandle n;

    ros::Publisher collision_pub = n.advertise<std_msgs::Bool>("sawyer_collision_avoidance/collision_state", 1000);

    CollisionGeometryPtr_t capsuleGeometry(new fcl::Capsuled(0.08, 0.28));
    fcl::Transform3d capsuleTransform(fcl::Translation3d(fcl::Vector3d(0., 0., 0.4)));
    auto pCapsule = new fcl::CollisionObjectd(capsuleGeometry, capsuleTransform);
    auto pSawyerGripper = new SawyerGripper(pCapsule);
    ros::Subscriber sawyer_endpoint_sub = n.subscribe("/robot/limb/right/endpoint_state", 1000, &SawyerGripper::on_sawyer_endpoint_msg, pSawyerGripper);

    CollisionGeometryPtr_t boxGeometry(new fcl::Boxd(1.2, 0.9, 0.2));
    fcl::Transform3d boxTransform(fcl::Translation3d(fcl::Vector3d(0.6, 0., -0.1)));
    auto box = new fcl::CollisionObjectd(boxGeometry, boxTransform);

    ros::Rate loop_rate(10);

    while (ros::ok()) {
        std_msgs::Bool msg;

        fcl::CollisionRequestd request;
        fcl::CollisionResultd result;
        fcl::collide(pSawyerGripper->getCollisionObj(), box, request, result);

        if (result.isCollision()) {
            msg.data = true;
            collision_pub.publish(msg);
        } else {
            msg.data = false;
        }

        collision_pub.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}


//
// Created by hejia on 6/29/18.
//

#include "sawyer_gripper.h"

SawyerGripper::~SawyerGripper() {

}

void SawyerGripper::on_sawyer_endpoint_msg(const intera_core_msgs::EndpointStateConstPtr& msg)
{
    fcl::Quaterniond orientation{msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z};
    fcl::Vector3d position{msg->pose.position.x, msg->pose.position.y, msg->pose.position.z};
    this->m_pCollisionObj->setTransform(orientation, position);
}

fcl::CollisionObjectd* SawyerGripper::getCollisionObj() {
    return m_pCollisionObj;
}
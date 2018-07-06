//
// Created by hejia on 6/29/18.
//

#ifndef SAWYER_COLLISION_AVOIDANCE_SAWYER_GRIPPER_H
#define SAWYER_COLLISION_AVOIDANCE_SAWYER_GRIPPER_H

#include <fcl.h>
#include <ros/ros.h>
#include <intera_core_msgs/EndpointState.h>

class SawyerGripper {
public:
    SawyerGripper(fcl::CollisionObjectd* pCollisionObj) : m_pCollisionObj(pCollisionObj) {

    }

    virtual ~SawyerGripper();

private:
    fcl::CollisionObjectd* m_pCollisionObj;

public:
    void on_sawyer_endpoint_msg(const intera_core_msgs::EndpointStateConstPtr& msg);

    fcl::CollisionObjectd* getCollisionObj();
};

#endif //SAWYER_COLLISION_AVOIDANCE_SAWYER_GRIPPER_H

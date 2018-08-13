import copy

from geometry_msgs.msg import Point, Quaternion, TransformStamped
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import rospy

TEST_VICON_TOPIC = 'vicon/sawyer_block/sawyer_block'

LOW_PASS_ALPHA = 0.1

OBSERVED_TRANSLATIONS = []

OBSERVED_ROTATIONS = []

SMOOTHED_TRANSLATIONS = []

SMOOTHED_TRANSLATION = Point()

SMOOTHED_ROTATIONS = []

SMOOTHED_ROTATION = Quaternion()

INITIALIZED = False


def vicon_update(data):
    global INITIALIZED
    global SMOOTHED_ROTATION
    global SMOOTHED_ROTATIONS
    global SMOOTHED_TRANSLATION
    global SMOOTHED_TRANSLATIONS

    observed_translation = data.transform.translation
    observed_rotation = data.transform.rotation

    if not INITIALIZED:
        SMOOTHED_TRANSLATION = observed_translation
        SMOOTHED_ROTATION = observed_rotation

        INITIALIZED = True
    else:
        SMOOTHED_TRANSLATION.x = SMOOTHED_TRANSLATION.x + LOW_PASS_ALPHA * (
            observed_translation.x - SMOOTHED_TRANSLATION.x)
        SMOOTHED_TRANSLATION.y = SMOOTHED_TRANSLATION.y + LOW_PASS_ALPHA * (
            observed_translation.y - SMOOTHED_TRANSLATION.y)
        SMOOTHED_TRANSLATION.z = SMOOTHED_TRANSLATION.z + LOW_PASS_ALPHA * (
            observed_translation.z - SMOOTHED_TRANSLATION.z)
        SMOOTHED_ROTATION.x = SMOOTHED_ROTATION.x + LOW_PASS_ALPHA * (
            observed_rotation.x - SMOOTHED_ROTATION.x)
        SMOOTHED_ROTATION.y = SMOOTHED_ROTATION.y + LOW_PASS_ALPHA * (
            observed_rotation.y - SMOOTHED_ROTATION.y)
        SMOOTHED_ROTATION.z = SMOOTHED_ROTATION.z + LOW_PASS_ALPHA * (
            observed_rotation.z - SMOOTHED_ROTATION.z)
        SMOOTHED_ROTATION.w = SMOOTHED_ROTATION.w + LOW_PASS_ALPHA * (
            observed_rotation.w - SMOOTHED_ROTATION.w)

    SMOOTHED_TRANSLATIONS.append(copy.deepcopy(SMOOTHED_TRANSLATION))
    SMOOTHED_ROTATIONS.append(copy.deepcopy(SMOOTHED_ROTATION))

    OBSERVED_TRANSLATIONS.append(observed_translation)
    OBSERVED_ROTATIONS.append(observed_rotation)

    print('Currently has {}'.format(len(OBSERVED_ROTATIONS)))

    if len(OBSERVED_ROTATIONS) == 1500:
        rospy.signal_shutdown('Sampleing is enough!')


def run():
    rospy.init_node('demo_smooth_vicon_data', anonymous=True)

    rospy.Subscriber(TEST_VICON_TOPIC, TransformStamped, vicon_update)

    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        r.sleep()

    # vis
    # obs_rots_y = []
    # for obs_rot in OBSERVED_ROTATIONS:
    #     obs_rots_y.append(obs_rot.y)
    #
    # smoothed_rots_y = []
    # for smoothed_rot in SMOOTHED_ROTATIONS:
    #     smoothed_rots_y.append(smoothed_rot.y)
    obs_trans_y = []
    for obs_tran in OBSERVED_TRANSLATIONS:
        obs_trans_y.append(obs_tran.y)

    smoothed_trans_y = []
    for smoothed_tran in SMOOTHED_TRANSLATIONS:
        smoothed_trans_y.append(smoothed_tran.y)

    t = np.linspace(0, 1500, 1500)

    plt.plot(t, obs_trans_y, 'r')
    plt.plot(t, smoothed_trans_y, 'b')

    plt.show()


if __name__ == '__main__':
    run()

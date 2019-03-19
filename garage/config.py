import os
import os.path as osp

# General config
GARAGE_PROJECT_PATH = os.environ.get(
    'GARAGE_PROJECT_PATH', osp.abspath(osp.join(osp.dirname(__file__), '..')))
GARAGE_LOG_DIR = os.environ.get('GARAGE_LOG_DIR',
                                osp.join(GARAGE_PROJECT_PATH, 'data'))
GARAGE_LOG_TENSORBOARD = bool(os.environ.get('GARAGE_LOG_TENSORBOARD', True))
GARAGE_USE_TF = bool(os.environ.get('GARAGE_USE_TF', False))
GARAGE_USE_GPU = bool(os.environ.get('GARAGE_USE_GPU', False))
GARAGE_MUJOCO_KEY_PATH = os.environ.get('GARAGE_MUJOCO_KEY_PATH',
                                        osp.expanduser('~/.mujoco'))
GARAGE_ENV = eval(os.environ.get('GARAGE_ENV', '{}'))
GARAGE_LABEL = os.environ.get('GARAGE_LABEL', 'default')

# Code copying rules (for Docker/AWS/Kubernetes)
GARAGE_CODE_SYNC_IGNORES = eval(
    os.environ.get(
        'GARAGE_CODE_SYNC_IGNORES', '''[
            "*.git/*",
            "*data/*",
            "*src/*",
            "*.pods/*",
            "*tests/*",
            "*examples/*",
            "docs/*"
        ]'''))
GARAGE_FAST_CODE_SYNC = bool(os.environ.get('GARAGE_FAST_CODE_SYNC', True))
GARAGE_FAST_CODE_SYNC_IGNORES = eval(
    os.environ.get('GARAGE_FAST_CODE_SYNC_IGNORES',
                   '[".git", "data", ".pods"]'))

GARAGE_DOCKER_IMAGE = os.environ.get('GARAGE_DOCKER_IMAGE',
                                     'rlworkgroup/garage-headless')
GARAGE_DOCKER_LOG_DIR = os.environ.get('GARAGE_DOCKER_LOG_DIR', '/tmp/expt')
GARAGE_DOCKER_CODE_DIR = os.environ.get('GARAGE_DOCKER_CODE_DIR',
                                        '/root/code/garage')

# AWS
GARAGE_AWS_S3_PATH = os.environ.get('GARAGE_AWS_S3_PATH', 'INVALID_S3_PATH')
GARAGE_AWS_IMAGE_ID = os.environ.get('GARAGE_AWS_IMAGE_ID', None)
GARAGE_AWS_INSTANCE_TYPE = os.environ.get('GARAGE_AWS_INSTANCE_TYPE',
                                          'm4.xlarge')
GARAGE_AWS_KEY_NAME = os.environ.get('GARAGE_AWS_KEY_NAME', None)
GARAGE_AWS_SPOT = bool(os.environ.get('GARAGE_AWS_SPOT', True))
GARAGE_AWS_SPOT_PRICE = os.environ.get('GARAGE_AWS_SPOT_PRICE', '1.0')
GARAGE_AWS_ACCESS_KEY = os.environ.get("GARAGE_AWS_ACCESS_KEY", None)
GARAGE_AWS_ACCESS_SECRET = os.environ.get("GARAGE_AWS_ACCESS_SECRET", None)
GARAGE_AWS_IAM_INSTANCE_PROFILE_NAME = os.environ.get(
    'GARAGE_AWS_IAM_INSTANCE_PROFILE_NAME', 'garage')
GARAGE_AWS_SECURITY_GROUPS = eval(
    os.environ.get('GARAGE_AWS_SECURITY_GROUPS', '["garage"]'))
GARAGE_AWS_SECURITY_GROUP_IDS = eval(
    os.environ.get('GARAGE_AWS_SECURITY_GROUP_IDS', '[]'))
GARAGE_AWS_NETWORK_INTERFACES = eval(
    os.environ.get('GARAGE_AWS_NETWORK_INTERFACES', '[]'))
GARAGE_AWS_EXTRA_CONFIGS = eval(
    os.environ.get('GARAGE_AWS_EXTRA_CONFIGS', '{}'))
GARAGE_AWS_REGION_NAME = os.environ.get('GARAGE_AWS_REGION_NAME', 'us-east-1')
GARAGE_AWS_CODE_SYNC_S3_PATH = os.environ.get('GARAGE_AWS_CODE_SYNC_S3_PATH',
                                              's3://to/be/overridden')
GARAGE_AWS_EBS_OPTIMIZED = bool(
    os.environ.get('GARAGE_AWS_EBS_OPTIMIZED', True))

# Kubernetes
GARAGE_KUBE_DEFAULT_RESOURCES = eval(
    os.environ.get('GARAGE_KUBE_DEFAULT_RESOURCES',
                   '{"requests": {"cpu": 0.8}}'))
GARAGE_KUBE_DEFAULT_NODE_SELECTOR = eval(
    os.environ.get('GARAGE_KUBE_DEFAULT_NODE_SELECTOR',
                   '{"aws/type": "m4.xlarge"}'))
GARAGE_KUBE_PREFIX = os.environ.get('GARAGE_KUBE_PREFIX', 'garage_')
GARAGE_KUBE_POD_DIR = os.environ.get('GARAGE_KUBE_POD_DIR',
                                     osp.join(GARAGE_PROJECT_PATH, '/.pods'))

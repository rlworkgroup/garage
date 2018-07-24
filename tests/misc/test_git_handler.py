import time

from git.exc import BadName, InvalidGitRepositoryError, NoSuchPathError
from termcolor import colored

from garage.misc.git_handler import GitHandler


def time_str():
    return str(int(time.time()))


def error_init_git_handler(exception, git_hdlr_arg):
    error_found = False

    try:
        git_handler = GitHandler(**git_hdlr_arg)
    except exception:
        error_found = True

    assert error_found, (
        "Argument %s in GitHandler must throw %s" % (git_hdlr_arg, exception))


def test_init_git_handler():
    invalid_dir = "/invalid/dir"
    invalid_url = "https://github.com/invalid/repo.git"
    not_git_dir = "../"

    invalid_dir_arg = {"work_directory": "/invalid/dir"}
    error_init_git_handler(NoSuchPathError, invalid_dir_arg)

    not_git_dir_arg = {"work_directory": "../"}
    error_init_git_handler(InvalidGitRepositoryError, not_git_dir_arg)

    invalid_url_arg = {"repo_url": "https://github.com/invalid/repo.git"}
    error_init_git_handler(AssertionError, invalid_url_arg)


def verify_branch(branch_name, branch_exists, git_handler, sha_to_verify=""):
    """Verify whether the branch exits or not.

    If branch_exists is false, then it's verified that the branch identified by
    branch_name does not exist, and it's verified it exists otherwise.
    """
    confirm_branch = branch_exists

    try:
        branch_sha = git_handler.get_branch_sha(branch_name)
    except BadName:
        confirm_branch = not confirm_branch

    if branch_exists:
        assert confirm_branch, ("Branch %s must exist." % branch_name)
        if sha_to_verify:
            assert sha_to_verify == branch_sha, \
                    ("Branch %s is not matching the right sha %s"
                     % (branch_name, sha_to_verify))
    else:
        assert confirm_branch, ("Branch %s must not exist." % branch_name)


def test_branch_sha():
    git_handler = GitHandler()

    invalid_remote_branch = "origin_" + time_str() + "/master"
    verify_branch(invalid_remote_branch, False, git_handler)

    valid_remote_branch = "origin/master"
    verify_branch(valid_remote_branch, True, git_handler)

    # Verify a branch does not exist
    local_branch = "branch_" + time_str()
    verify_branch(local_branch, False, git_handler)

    # Create a branch and verify it exist
    git_handler.create_branch(local_branch)
    verify_branch(local_branch, True, git_handler)
    git_handler.delete_branch(local_branch)

    # Create a local branch in a specific reference and verify its SHA
    sha_on_master = "5c5f63674fe6a39125f0bca1e35a01e2a53f6637"
    local_branch = "branch_" + time_str()
    git_handler.create_branch(local_branch, sha_on_master)
    verify_branch(local_branch, True, git_handler, sha_on_master)
    git_handler.delete_branch(local_branch)

    # Retrieve the name of the current branch if there's a branch currently
    # checked out or the SHA where the detached HEAD is before switching
    current_branch = git_handler.get_current_branch_name()
    if not current_branch:
        current_sha = git_handler.get_sha()
    # Create and switch to a branch, and verify its name
    local_branch = "branch_" + time_str()
    git_handler.checkout_new_branch(local_branch)
    retrieved_branch_name = git_handler.get_current_branch_name()
    assert local_branch == retrieved_branch_name, \
            "The branch %s must be checked out" % local_branch
    if current_branch:
        git_handler.checkout(current_branch)
    else:
        git_handler.reset(current_sha)
    git_handler.delete_branch(local_branch)


def verify_tag(tag_name, tag_exists, git_handler, sha_to_verify=""):
    """Verify whether the tag exits or not.

    If tag_exists is false, then it's verified that the tag identified by
    tag_name does not exist, and it's verified it exists otherwise.
    """
    confirm_tag = tag_exists

    try:
        tag_sha = git_handler.get_tag_sha(tag_name)
    except BadName:
        confirm_tag = not confirm_tag

    if tag_exists:
        assert confirm_tag, ("Tag %s must exist." % tag_name)
        if sha_to_verify:
            assert sha_to_verify == tag_sha, \
                    ("Tag %s is not matching the right sha %s"
                     % (tag_name, sha_to_verify))
    else:
        assert confirm_tag, ("Tag %s must not exist." % tag_name)


def test_tag_sha():
    git_handler = GitHandler()

    # Verify that a tag is not in repository
    invalid_tag = "tag_" + time_str()
    verify_tag(invalid_tag, False, git_handler)

    # Create a tag in a specific reference and verify its SHA
    tag_name = "tag_" + time_str()
    sha_on_master = "5378034533a47dad101c6f60e628a64442b439ed"
    git_handler.create_tag(tag_name, sha_on_master)
    verify_tag(tag_name, True, git_handler, sha_on_master)
    git_handler.delete_tag(tag_name)


def test_sha():
    git_handler = GitHandler()

    head_sha = git_handler.get_sha()
    assert head_sha, "The SHA of the HEAD must be returned with get_sha()"

    # Verify a real SHA
    sha_to_verify = "cc5d4627a7306c3586257f29c2def6121d7f04ec"
    returned_sha = git_handler.get_sha(sha_to_verify)
    assert sha_to_verify == returned_sha, ("The SHA returned by get_sha() " \
                                           "is unexpected")

    # Verify a short real SHA
    short_sha_to_verify = "dfbe8ae2"
    returned_sha = git_handler.get_sha(short_sha_to_verify)
    assert returned_sha.startswith(short_sha_to_verify), \
            "The SHA returned by get_sha() is unexpected"

    # Verify a fake SHA
    sha_to_verify = "779fea67d6ab6c8085507c71697bd27d15835b91"
    fake_sha = False
    try:
        returned_sha = git_handler.get_sha(sha_to_verify)
        print(returned_sha)
    except ValueError:
        fake_sha = True
    assert fake_sha, ("The SHA returned by get_sha() is unexpected")


def test_verify_reference():
    git_handler = GitHandler()

    # Verify that an invalid key in the passed reference is invalid
    key_value = "invalid_key: invalid_value"
    invalid_key = False
    try:
        git_handler._verify_reference(key_value)
    except KeyError:
        invalid_key = True
    assert invalid_key, "The key-value %s must be invalid" % key_value

    # Verify that a tag, branch and sha do not exist using a key-value as input
    tag_name = "tag_" + time_str()
    invalid_reference = False
    try:
        git_handler._verify_reference("tag: " + tag_name)
    except BadName:
        invalid_reference = True
    assert invalid_reference, "The tag %s must be invalid" % tag_name

    branch_name = "branch_" + time_str()
    invalid_reference = False
    try:
        git_handler._verify_reference("branch: " + branch_name)
    except BadName:
        invalid_reference = True
    assert invalid_reference, "The branch %s must be invalid" % branch_name

    invalid_sha = "5730843353" + time_str()
    invalid_reference = False
    try:
        git_handler._verify_reference("sha: " + invalid_sha)
    except BadName:
        invalid_reference = True
    assert invalid_reference, "The sha %s must be invalid" % invalid_sha

    # Create tag and branch on a specific SHA in repository, and then verify
    # the key-value for the tag, branch and sha
    tag_name = "tag_" + time_str()
    sha_on_master = "5378034533a47dad101c6f60e628a64442b439ed"
    git_handler.create_tag(tag_name, sha_on_master)
    sha_retrieved = git_handler._verify_reference("tag: " + tag_name)
    assert sha_on_master == sha_retrieved, \
            "The tag %s must be valid" % tag_name
    git_handler.delete_tag(tag_name)

    branch_name = "branch_" + time_str()
    sha_on_master = "5378034533a47dad101c6f60e628a64442b439ed"
    git_handler.create_branch(branch_name, sha_on_master)
    sha_retrieved = git_handler._verify_reference("branch: " + branch_name)
    assert sha_on_master == sha_retrieved, \
            "The branch %s must be invalid" % branch_name
    git_handler.delete_branch(branch_name)

    sha_on_master = "5378034533a47dad101c6f60e628a64442b439ed"
    sha_retrieved = git_handler._verify_reference("sha: " + sha_on_master)
    assert sha_on_master == sha_retrieved, \
            "The SHA %s must be invalid" % sha_on_master


def test_switch_and_restore():
    # This test is basically the same as the method run_experiment_on_ref
    # except for calling run_experiment to avoid running any training.
    git_handler = GitHandler()
    sha_on_master = "5378034533a47dad101c6f60e628a64442b439ed"
    ref_sha = git_handler._verify_reference("sha: " + sha_on_master)
    restore_sha = git_handler.get_sha()
    stash_sha = git_handler.create_stash()
    branch_name = git_handler.get_current_branch_name()
    ref_branch_name = "run_exp_on_" + time_str()
    try:
        git_handler._switch_to_ref(ref_branch_name, ref_sha)
        current_sha = git_handler.get_sha()
        # Once the working directory is switched, it's asserted that the HEAD
        # is at the reference based on the SHA
        assert current_sha == ref_sha, \
                "The working area must be at SHA %s" % ref_sha
    except BaseException:
        print(colored(
            "If the working area is not correctly restored after an " \
            "unexpected error, run\nthe following command(s) to restore " \
            "it:", "yellow"))
        if branch_name:
            print(colored("$ git checkout %s" % (branch_name), "yellow"))
        else:
            print(colored("$ git reset --hard %s" % (restore_sha), "yellow"))

        if stash_sha:
            print(colored("$ git stash apply %s" % (stash_sha), "yellow"))
        raise
    finally:
        git_handler._restore_working_dir(ref_branch_name, stash_sha,
                                         branch_name, restore_sha)
        current_sha = git_handler.get_sha()
        # Once the working directory is restored, it's asserted that the HEAD
        # is at the right commit based on the SHA
        assert current_sha == restore_sha, \
                "The working area must be at SHA %s" % restore_sha

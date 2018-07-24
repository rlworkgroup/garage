"""Perform different git tasks while running experiments."""
import importlib
import signal
import time

from git.exc import BadName, GitCommandError
from termcolor import colored

from garage.config import GIT_REPO_URL
from garage.config import PROJECT_PATH
from git import Repo


class GitHandler:
    """Execute diverse git commands.

    This class makes use of the sh package to execute git commands just like
    in the terminal shell, with the convenience of grouping commands to perform
    more complex tasks.

    Preconditions of this class:
        - The working directory is not in the middle of a conflict
        resolution.
        - No branch is named HEAD.
    """

    def __init__(self, work_directory=PROJECT_PATH, repo_url=GIT_REPO_URL):
        """Verify the git working directory.

        It's verified that the working directory exists and contains a
        git working directory by obtaining the name of the remotes.
        Then, it's checked that one of the remotes points to the garage
        repository by checking the repository URL.

        Parameters
        ----------
        - work_directory: working directory of the git repository
        - repo_url: URL of the git repository found in the working directory

        """
        self.work_dir = work_directory
        self.valid_targets = {
            "branch": self.get_branch_sha,
            "sha": self.get_sha,
            "tag": self.get_tag_sha
        }
        self.repo = Repo(work_directory)
        remote_found = self._check_url(repo_url)
        assert remote_found, colored("The garage remote couldn't be found " \
                                     "in the git working directory %s" %
                                     self.work_dir, "red")

    def _check_url(self, repo_url):
        """Return true if one of the remotes points to the repository URL."""
        valid_url = False
        for remote in self.repo.remotes:
            for remote_url in remote.urls:
                if remote_url == repo_url:
                    valid_url = True
                    break
        return valid_url

    def create_branch(self, branch_name, target_ref="HEAD"):
        """Create a branch on the target reference."""
        self.repo.git.branch(branch_name, target_ref)

    def checkout_new_branch(self, branch_name, target_ref="HEAD"):
        """Create a branch on the target reference."""
        self.repo.git.checkout(branch_name, target_ref, b=True)

    def checkout(self, branch_name):
        """Checkout a branch."""
        self.repo.git.checkout(branch_name)

    def delete_branch(self, branch_name):
        """Create a branch on the target reference."""
        self.repo.git.branch(branch_name, D=True)

    def get_branch_sha(self, branch_name):
        """Return the 40 character SHA of the branch name.

        The name can be that of a local or remote branch.
        """
        # Local branches may contain slashes just like remote branches
        remote_branch = False
        sha = ""
        for remote in self.repo.remotes:
            remote_name = remote.name + "/"
            if branch_name.startswith(remote_name):
                remote_branch = True
                break

        if remote_branch:
            full_branch_name = "refs/remotes/" + branch_name
        else:
            full_branch_name = "refs/heads/" + branch_name

        return self.get_sha(full_branch_name)

    def create_tag(self, tag_name, target_ref="HEAD"):
        """Create a tag on the target reference."""
        self.repo.git.tag(tag_name, target_ref)

    def delete_tag(self, tag_name):
        """Create a tag on the target reference."""
        self.repo.git.tag(tag_name, d=True)

    def get_tag_sha(self, tag_name):
        """Return the 40 character SHA of the tag name."""
        full_tag_name = "refs/tags/" + tag_name
        return self.get_sha(full_tag_name)

    def reset(self, reference="HEAD"):
        """Reset the working directory to the indicated reference."""
        self.repo.git.reset("--hard", reference)

    def get_sha(self, reference="HEAD"):
        """Return the 40 character SHA of the reference.

        The reference can be a local branch ("refs/heads/<branch_name>"), a
        remote branch ("refs/remotes/<branch_name>"), a tag
        ("refs/tags/<tag_name>"), HEAD or a SHA.
        Passing a SHA as reference comes useful to verify the SHA is valid.
        HEAD is just a reference to the commit where the working directory is
        currently checked out.

        If the reference is not valid, the exception sh.ErrorReturnCode_128 is
        thrown.
        """
        try:
            commit = self.repo.rev_parse(reference)
        except (BadName, ValueError):
            print(
                colored("The reference %s does not exist", "red") % reference)
            raise
        return str(commit)

    def create_stash(self):
        """Stashes the local changes that have not been staged.

        The stash is not stored anywhere in the ref namespace so the user does
        not have remove it manually later.

        Returns
        -------
        The 40 character SHA if local changes were found, or an empty string
        otherwise.

        """
        stash_sha = ""
        try:
            self.repo.git.diff(exit_code=True)
        except GitCommandError:
            # Exit code 1 means there are differences
            stash_sha = self.repo.git.stash("create")
            pass
        return stash_sha

    def get_current_branch_name(self):
        """Return the name of the checked out branch.

        If the HEAD is in detached state, an empty string is returned.

        Precondition
        ------------
        - No branch (local or remote) in the repository is named "HEAD".

        """
        curr_branch_name = self.repo.git.rev_parse(
            self.repo.head, abbrev_ref=True)
        if curr_branch_name == "HEAD":
            curr_branch_name = ""
        return curr_branch_name

    def run_experiment_on_target(self, target, exp_args):
        """Run a experiment on the indicated commit defined by target.

        After the target is verified, if there are any non-staged changes in
        the working directory, they're stashed, the target commit is checked
        out, the experiment is executed and the working directory is restored
        at the end, including the stashed changes.

        Parameters
        ----------
        - target: a key-value string, where the key indicates the type of
            target (tag, branch, sha), and the value the reference for the
            corresponding type. For example:
                sha: 8d54s23a
                branch: my_branch
                branch: remote/master
                tag: my_tag
        - exp_args: it's a dictionary of arguments obtained from the function
            run_experiment at garage.misc.instrument, that will be passed to
            the experiment once the target is checked out.

        Precondition
        ------------
        - The working directory is not in the middle of a conflict resolution.

        """
        assert isinstance(target, str), ("The git target has to be a string")
        self.target = target
        self.target_sha = self._verify_target(self.target)
        self.restore_sha = self.get_sha()
        self.stash_sha = self.create_stash()
        self.branch_name = self.get_current_branch_name()
        self.target_branch_name = ("run_exp_on_" + str(int(time.time())))
        try:
            self._switch_to_target(self.target_branch_name, self.target_sha)
            current_sha = self.get_sha()
            self._run_experiment(exp_args)
        except BaseException:
            print(colored(
                "If the working area is damaged after " \
                "an unexpected error while trying to\nswitch to the " \
                "target in the repository, run the following command(s) " \
                "to\nrestore it:", "yellow"))
            if self.branch_name:
                print(
                    colored("$ git checkout %s" % (self.branch_name),
                            "yellow"))
            else:
                print(
                    colored("$ git reset --hard %s" % (self.restore_sha),
                            "yellow"))

            if self.stash_sha:
                print(
                    colored("$ git stash apply %s" % (self.stash_sha),
                            "yellow"))
            raise
        finally:
            self._restore_working_dir(self.target_branch_name, self.stash_sha,
                                      self.branch_name, self.restore_sha)

    def _verify_target(self, target):
        """Verify the key-value target is valid.

        Parameters
        ----------
        - target: a key-value string, where the key indicates the type of
            target (tag, branch, sha), and the value the reference for the
            corresponding type. For example:
                sha: 8d54s23a
                branch: my_branch
                branch: remote/master
                tag: my_tag

        Returns
        -------
        If the target is valid, the corresponding SHA.

        """
        target = [target.strip() for target in target.split(":")]
        sha = ""
        assert (len(target) == 2 and target[0]
                and target[1]), self._get_target_use_message()
        try:
            verify = self.valid_targets[target[0]]
            sha = verify(target[1])
        except KeyError:
            print(self._get_target_use_message())
            raise
        return sha

    def _get_target_use_message(self):
        """Print a use message if the user passed an invalid target."""
        return colored(
            "\nInvalid target: targets can be defined with a SHA, branch " \
            "name (local or remote) or tag\nas shown " \
            "in the following examples:\nsha: 8d54s23a" \
            "\nbranch: my_branch\nbranch: remote/master\ntag: my_tag\n",
            "yellow")

    def _switch_to_target(self, target_branch_name, target_sha):
        """Clean working directory and check out the target branch.

        The target branch is created during the checkout operation.
        SIGINT is temporarily blocked while performing the git operations to
        avoid corrupting the working directory.
        """
        # Block SIGINT to avoid ending in an unknown state
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        # Clean working directory
        self.reset()
        # Change to target directory using a new branch
        self.checkout_new_branch(target_branch_name, target_sha)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})

    def _run_experiment(self, exp_args):
        """Call method run_experiment at garage.misc.instrument.

        The git target is removed from the dictionary of arguments passed to
        run_experiment, and the key arguments are updated directly into the
        run_experiment arguments.

        Parameters
        ----------
        - exp_args: it's a dictionary of arguments obtained from the function
            run_experiment at garage.misc.instrument, that will be passed to
            the experiment once the target is checked out.

        """
        # Imports are here to avoid circular dependencies within garage.misc
        import garage.misc.instrument
        from garage.misc import instrument
        if "git_target" in exp_args:
            exp_args.pop("git_target")
        if "kwargs" in exp_args:
            kwargs = exp_args["kwargs"]
            exp_args.pop("kwargs")
            exp_args.update(kwargs)
        importlib.reload(garage.misc.instrument)
        instrument.run_experiment(**exp_args)

    def _restore_working_dir(self,
                             target_branch_name,
                             stash_sha="",
                             branch_name="",
                             restore_sha=""):
        """Restore the working directory once the experiment is done.

        Any stashed changes are applied as well.
        SIGINT is temporarily blocked while performing the git operations to
        avoid corrupting the working directory.
        """
        # Block SIGINT to avoid ending in an unknown state
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        if branch_name:
            self.checkout(branch_name)
        else:
            self.reset(restore_sha)
        if stash_sha:
            self.repo.git.stash("apply", stash_sha)
        self.delete_branch(target_branch_name)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})

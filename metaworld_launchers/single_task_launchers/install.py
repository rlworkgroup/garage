import pip
package = f'metaworld @ https://git@api.github.com/repos/rlworkgroup/metaworld/tarball/new-reward-functions'
pip.main(['uninstall', '--yes', "metaworld"])
pip.main(['install', '--ignore-installed', package])
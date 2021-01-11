import subprocess

mt1_pick_place = [
        'mtppo_metaworld_mt1_env_name=pick-place-v2_seed=3343_epochs=500_batch_size=25000',
        'mtppo_metaworld_mt1_env_name=pick-place-v2_seed=6236_epochs=500_batch_size=25000',
        'mtppo_metaworld_mt1_env_name=pick-place-v2_seed=9525_epochs=500_batch_size=25000',
        'maml_trpo_metaworld_mt1_env_name=pick-place-v2_seed=820_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'maml_trpo_metaworld_mt1_env_name=pick-place-v2_seed=7769_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'maml_trpo_metaworld_mt1_env_name=pick-place-v2_seed=5751_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'maml_ppo_metaworld_mt1_env_name=pick-place-v2_seed=9591_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'maml_ppo_metaworld_mt1_env_name=pick-place-v2_seed=3448_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'maml_ppo_metaworld_mt1_env_name=pick-place-v2_seed=1342_epochs=1000_rollouts_per_task=10_meta_batch_size=20',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=2219_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=6486_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=6698_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        ]

for file in mt1_pick_place:
#     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
    subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/mt1-pick-place/{file}/progress.csv {file}.csv"], shell=True)

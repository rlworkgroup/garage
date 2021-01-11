import subprocess

ml1_pick_place = [
        'maml_trpo_metaworld_ml1_pick_place_seed=2939',
        'maml_trpo_metaworld_ml1_pick_place_seed=4583',
        'maml_trpo_metaworld_ml1_pick_place_seed=7612',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=3735_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=6687_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=pick-place-v2_seed=7899_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        ]

ml1_push = [
        'maml_trpo_metaworld_ml1_push_seed=3905',
        'maml_trpo_metaworld_ml1_push_seed=5248',
        'maml_trpo_metaworld_ml1_push_seed=7082',
        'rl2_ppo_metaworld_ml1_env_name=push-v2_seed=1376_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=push-v2_seed=4702_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        'rl2_ppo_metaworld_ml1_env_name=push-v2_seed=5314_meta_batch_size=10_n_epochs=2000_episode_per_task=10',
        ]

# for file in mt10:
# #     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
#     subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/mt10/{file}/progress.csv {file}.csv"], shell=True)

for file in ml1_pick_place:
#     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
    subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/ml1-pick-place/{file}/progress.csv {file}.csv"], shell=True)

for file in ml1_push:
#     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
    subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/ml1-push/{file}/progress.csv {file}.csv"], shell=True)

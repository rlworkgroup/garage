import subprocess

ml10 = [
        'maml_trpo_metaworld_mt10_seed=838_il=0.05_extra_tags=none',
        'maml_trpo_metaworld_mt10_seed=2332_il=0.05_extra_tags=none',
        'maml_trpo_metaworld_mt10_seed=5326_il=0.05_extra_tags=none',
        'mttrpo_metaworld_mt10_seed=3813_epochs=2000_batch_size=5000_n_workers=10_n_tasks=10',
        'mttrpo_metaworld_mt10_seed=6160_epochs=2000_batch_size=5000_n_workers=10_n_tasks=10',
        'mttrpo_metaworld_mt10_seed=826_epochs=2000_batch_size=5000_n_workers=10_n_tasks=10',
        ]

# for file in mt10:
# #     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
#     subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/mt10/{file}/progress.csv {file}.csv"], shell=True)

for file in ml10:
#     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
    subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/mt10_maml_vs_mttrpo/{file}/progress.csv {file}.csv"], shell=True)

# seeds = [7166, 902, 8946, 9227, 6267]
# seeds = [8516, 3352, 5592, 2181, 4322]
# for seed in seeds:
#         try:
#                 subprocess.run([f'gsutil -m mv -r gs://metaworld-v2-paper-results/ml10/maml_trpo_metaworld_mt10_seed={seed}/rl2_ppo_metaworld_ml10_seed={seed}_meta_batch_size=10_n_epochs=400_episode_per_task=10/*  gs://metaworld-v2-paper-results/ml10/rl2_ppo_metaworld_ml10_seed={seed}/'])
#         except:
#                 continue

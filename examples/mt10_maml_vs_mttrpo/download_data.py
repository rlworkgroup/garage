import subprocess

MT10_MAML_RERUN = [
        'maml_trpo_metaworld_mt10_seed=1312_il=0.05_extra_tags=none',
        'maml_trpo_metaworld_mt10_seed=1483_il=0.05_extra_tags=none',
        'maml_trpo_metaworld_mt10_seed=674_il=0.05_extra_tags=none',
]


for file in MT10_MAML_RERUN:
#     subprocess.run([f"gsutil", "cp", f"gs://metaworld-v2-paper-results/mt10/{file}/progress.csv", f"{file}.csv"], shell=True)
    subprocess.run([f"gsutil cp gs://metaworld-v2-paper-results/mt10_maml_vs_mttrpo/{file}/progress.csv {file}.csv"], shell=True)

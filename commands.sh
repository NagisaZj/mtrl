# 2 6 9 grad
# 4 10 soft
# 7 8 5 imitate

#new
# 13 14 15  grad  19 small
# 16 17 18  22 mine  20 small 21 50 test 23 50 test interval 50

#mt50
# mine: 21 50 test 23 50 test interval 50
# mine 24 25 26 27 evo test 32 33 34 35 smaller 36 37 ori 38 39 more ori
# pcgrad 28 29 30 31  40: 500 step
# mtsac


#path length 500
#pcgrad pp 40 41
#mine pp 42 43
#pcgrad mt10 44 45
#pcgrad mt50 46 47
#mtsac  mt10
#mtsac mt50 48 49
#mine mt10
#mine mt50


#mtsac
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=1 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=10 \
agent.multitask.should_use_disentangled_alpha=True \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=True \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=True

#mtmhsac
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt1 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=1 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
agent.multitask.should_use_disentangled_alpha=True \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_multi_head_policy=True \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False

#pp
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt1 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=43 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
agent.multitask.should_use_disentangled_alpha=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.encoder.type_to_select=identity \
setup.pseudo_thres=0.5 \
setup.pseudo_interval=500


CUDA_VISIBLE_DEVICES=5 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt10 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=45 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
agent.multitask.should_use_disentangled_alpha=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.encoder.type_to_select=identity \
setup.pseudo_thres=0.7 \
setup.pseudo_interval=250 \
setup.use_evo=1


CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt50 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=47 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
agent.multitask.should_use_disentangled_alpha=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.encoder.type_to_select=identity \
setup.pseudo_thres=0.7 \
setup.pseudo_interval=250 \
setup.use_evo=1

CUDA_VISIBLE_DEVICES=7 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt1 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=20 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=25 \
agent.multitask.should_use_disentangled_alpha=False \
agent.multitask.should_use_task_encoder=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.encoder.type_to_select=identity


CUDA_VISIBLE_DEVICES=0  OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt1 \
agent=state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=10 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=True \
agent.encoder.type_to_select=feedforward \
agent.multitask.actor_cfg.should_condition_model_on_task_info=True \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.multitask.actor_cfg.moe_cfg.should_use=True \
agent.multitask.actor_cfg.moe_cfg.mode=soft_modularization \
agent.multitask.should_use_multi_head_policy=False \
agent.encoder.feedforward.hidden_dim=50 \
agent.encoder.feedforward.num_layers=2 \
agent.encoder.feedforward.feature_dim=50 \
agent.actor.num_layers=4 \
agent.multitask.task_encoder_cfg.model_cfg.pretrained_embedding_cfg.should_use=False
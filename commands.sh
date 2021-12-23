# 2 6 9 grad
# 4 10 soft
# 7 8 5 imitate

#new
# 13 14 15 grad
# 16 17 18 mine
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python3 -u main.py \
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

CUDA_VISIBLE_DEVICES=5 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python3 -u main.py \
setup=metaworld \
env=metaworld-mt1 \
agent=pcgrad_state_sac \
experiment.num_eval_episodes=1 \
experiment.num_train_steps=2000000 \
setup.seed=18 \
replay_buffer.batch_size=1280 \
agent.multitask.num_envs=50 \
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
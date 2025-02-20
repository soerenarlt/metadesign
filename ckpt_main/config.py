data_dir = 'data_main/data'
traindata_prefix = 'shuffled_data_'
split_train = 99
src_token_dict = 'data_main/tok.json'
tgt_token_dict = 'data_main/tok.json'
task_dir = 'data_main'

out_dir = 'out-main'
eval_interval = 2048 
eval_iters = 4*64 # how many iters to run for each eval
log_interval = 10 # don't print too too often

always_save_checkpoint = False

wandb_log = False # log to weights and biases
wandb_project = 'quantum'
wandb_run_name = 'main'

src_len = 640 # length of sequence
tgt_len = 640

batch_size = 64
n_layer = 18
n_head = 8
n_embd = 512
dropout = 0.05

learning_rate = 1e-4 
max_iters = 500000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = learning_rate # minimum learning rate
beta1 = 0.9
beta2 = 0.999 

warmup_iters = 1000 # not super necessary potentially

compile = False # do not torch compile the model
init_from = 'scratch' # 'scratch' or 'resume'

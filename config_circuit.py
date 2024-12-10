data_dir = 'data_circuits/data'
traindata_prefix = 'combined_'
split_train = 99
src_token_dict = 'data_circuits/src_tok.json'
tgt_token_dict = 'data_circuits/tgt_tok.json'
task_dir = 'data_circuits'

out_dir = 'out-circuits'

eval_interval = 2048 
eval_iters = 4*64 # how many iters to run for each eval
log_interval = 10 # don't print too too often

always_save_checkpoint = False

wandb_log = True # log to weights and biases
wandb_project = 'quantum'
wandb_run_name = 'cqt' #you don't need to give a name

src_len = 128 # length of sequence
tgt_len = 128

batch_size = 128
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.05

learning_rate = 2e-4
max_iters = 500000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = learning_rate/1
beta1 = 0.9
beta2 = 0.999 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially

compile = False # do not torch compile the model
init_from = 'scratch' # 'scratch' or 'resume'

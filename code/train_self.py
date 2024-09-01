from header import *
from datasets import *
from model import *
from config import *
# import warnings
# warnings.simplefilter("error")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    # model configurations
    parser.add_argument('--imagebind_ckpt_path', type=str) # the path that stores the imagebind checkpoint
    parser.add_argument('--if_load_lora', action=argparse.BooleanOptionalAction)  # the path that stores the imagebind checkpoint
    parser.add_argument('--if_load_decoder', action=argparse.BooleanOptionalAction)
    parser.add_argument('--if_load_delta', action=argparse.BooleanOptionalAction)
    parser.add_argument('--lora_ckpt_path', type=str)  # the path that stores the imagebind checkpoint
    parser.add_argument('--vicuna_ckpt_path', type=str) # the path that stores the vicuna checkpoint
    parser.add_argument('--gemma_ckpt_path', type=str)  # the path that stores the gemma checkpoint
    parser.add_argument('--decoder_ckpt_path', type=str) # the delta parameters trained in stage 1
    parser.add_argument('--delta_ckpt_path', type=str)  # the delta parameters trained in stage 1
    parser.add_argument('--max_tgt_len', type=int) # the maximum sequence length
    parser.add_argument('--stage', type=int) # the maximum sequence length
    parser.add_argument('--data_path', type=str) # the maximum sequence length
    parser.add_argument('--image_root_path', type=str) # the maximum sequence length

    return parser.parse_args()

def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])

def build_directory(path):
    if os.path.exists(path):
        pass
    else: # recursively construct directory
        os.makedirs(path, exist_ok=True)

def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', 
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    train_data, train_iter, sampler = load_self_supervised_dataset(args)
    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()


    # begin to train
    pbar = tqdm(total= length)    # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        iter_every_epoch = 0
        #for batch, batch_sft in zip(train_iter,train_iter_sft):
        for batch in train_iter:
            iter_every_epoch += 1
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )
            del batch


            current_step += 1
            # torch.cuda.empty_cache()
            # if iter_every_epoch % 1000 == 0:
            #     agent.save_model(args['save_path'], 0)
        # save at the end of the training
        torch.distributed.barrier()
        if args['if_load_lora']:
            agent.save_model_gemma_2(args['save_path'], epoch_i)
        else:
            agent.save_model_gemma(args['save_path'], epoch_i)
if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['layers'] = [7,15,23,31]
    main(**args)

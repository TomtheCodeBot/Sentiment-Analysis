import sys
sys.path.append('../')
import os
if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import warnings
warnings.filterwarnings('ignore')
from fastNLP import Trainer,Tester
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
import fitlog
from torch import optim
from models.wrapper_model import SequenceGeneratorModel
from models.loss import *
from models.metrics import *
from models.pipeline import *
from models.utils import *
from models.T5ABSA import *


# fitlog.debug()
lr = 5e-5
n_epochs = 100
batch_size = 10
num_beams = 1
dataset_name = 'pengb/16res'
opinion_first = False
length_penalty = 1.0

decoder_type = 'avg_score'
t5_name = 't5-base'
use_encoder_mlp = 1

dataset = "laptop"

demo = False
model_path = "output_model/t5/LaptopACOS"

if __name__ =="__main__":

    demo = False
    if demo:
        cache_fn = f"caches/data_{t5_name}_{dataset_name}_{opinion_first}_demo.pt"
    else:
        cache_fn = f"caches/data_{t5_name}_{dataset_name}_{opinion_first}.pt"

    @cache_results(cache_fn, _refresh=False)
    def get_data():
        pipe = T5BPEABSAPipe(tokenizer=t5_name, opinion_first=opinion_first)
        data_bundle = pipe.process_from_file(f'LaptopACOS/json', demo=demo)
        return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid

    data_bundle, tokenizer, mapping2id,mapping2targetid = get_data()
    max_len = 10
    max_len_a = {
        'penga/14lap': 0.9,
        'penga/14res': 1,
        'penga/15res': 1.2,
        'penga/16res': 0.9,
        'pengb/14lap': 1.1,
        'pengb/14res': 1.2,
        'pengb/15res': 0.9,
        'pengb/16res': 1.2
    }[dataset_name]

    idtarget2map=inv_map = {v: k for k, v in mapping2targetid.items()}

    bos_token_id = 0  #
    eos_token_id = 1  #
    label_ids = list(mapping2id.values())
    vocab_size = len(tokenizer)


    model = T5Seq2SeqModel.build_model(t5_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                        copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False)
    print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
    model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                                eos_token_id=eos_token_id,
                                max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                                restricter=None)
    #model = torch.load("/home/student/Sentiment-Analysis/output_model/LaptopACOS/best_SequenceGeneratorModel_quad_f_2022-07-02-04-03-29-755703")
    if torch.cuda.is_available():
        # device = list([i for i in range(torch.cuda.device_count())])
        device = 'cuda'
    else:
        device = 'cpu'

    parameters = []
    params = {'lr':lr, 'weight_decay':1e-2}
    params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
    parameters.append(params)

    params = {'lr':lr, 'weight_decay':1e-2}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    params = {'lr':lr, 'weight_decay':0}
    params['params'] = []
    for name, param in model.named_parameters():
        if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
            params['params'].append(param)
    parameters.append(params)

    optimizer = optim.AdamW(parameters)
    
    callbacks = []
    callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
    callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
    callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))
    fitlog.set_log_dir('caches')

    sampler = None
    # sampler = ConstTokenNumSampler('src_seq_len', max_token=1000)
    sampler = BucketSampler(seq_len_field_name='src_seq_len')
    metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids), opinion_first=opinion_first)

    trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                      loss=Seq2SeqLoss(),
                      batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                      num_workers=2, n_epochs=n_epochs, print_every=1,
                      dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='quad_f',
                      validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                      callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                      test_sampler=SortedSampler('src_seq_len'), dev_batch_size=batch_size)
    trainer.train(load_best_model=True)

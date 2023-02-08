import numpy as np


def get_max_len_max_len_a(data_bundle, max_len=10):
    """
    :param data_bundle:
    :param max_len:
    :return:
    """
    max_len_a = -1
    for name, ds in data_bundle.iter_datasets():
        if name=='train':continue
        src_seq_len = np.array(ds.get_field('src_seq_len').content)
        tgt_seq_len = np.array(ds.get_field('tgt_seq_len').content)
        _len_a = round(max(np.maximum(tgt_seq_len - max_len+2, 0)/src_seq_len), 1)

        if _len_a>max_len_a:
            max_len_a = _len_a

    return max_len, max_len_a


def get_num_parameters(model):
    num_param = 0
    for name, param in model.named_parameters():
        num_param += np.prod(param.size())
    print(f"The number of parameters is {num_param}")
    
def translateResult(sentences,results,idtarget2map,tokenizer):
    converted_sentences=[]
    for i in sentences:
        lst = []
        for word in i.split(" "):
            bpes = tokenizer.tokenize(word, add_prefix_space=True)
            lst.extend(bpes)
        converted_sentences.append(" ".join(lst))
    output_mold = []
    len_lst = []
    cap = len(idtarget2map)+2
    def translateResultChunk(block,sentence):
        output = []
        prev = None
        lst = sentence.split(" ")
        for i in block:
            if i<cap:
                output.append(idtarget2map[i-2])
                continue
            if prev is not None:
                chunk = lst[prev:i-cap]
                text = ""
                for i in chunk:
                    if i[0]=="Ġ":
                        text =text +" "+i[1:]
                    else:
                        text =text +i
                output.append(text.strip())
                prev = None
            else:
                prev = i-cap-1
        return output
    for i in range(0,len(converted_sentences)):
        blocks=[]
        count=0
        cur_pair = []
        for o in results[i]:
            k = int(o)
            if k==0 or k==1:
                continue
            
            if k<cap:
                cur_pair.append(k)

                if count<2:
                    count+=1
                else:
                    if not(len(cur_pair) != 7 or cur_pair[0] > cur_pair[1] or cur_pair[4] > cur_pair[5]):
                        blocks.append(translateResultChunk(cur_pair,"NULL "+converted_sentences[i]).copy())
                    cur_pair = []
                    count=0
            else:
                cur_pair.append(k)
        output_mold.append(blocks.copy())
    return output_mold

def tokenize_sentence(sentences,tokenizer,device = "cpu"):
    output_mold = []
    len_lst = []
    for i in range (0,len(sentences)):
        added_sentence = "<<null>> "+sentences[i]
        raw_words = added_sentence.split(" ")
        word_bpes = [[tokenizer.bos_token_id]]
        for word in raw_words:
            bpes = tokenizer.tokenize(word, add_prefix_space=True)
            bpes = tokenizer.convert_tokens_to_ids(bpes)
            word_bpes.append(bpes)
        word_bpes.append([tokenizer.eos_token_id])
        output = list(chain(*word_bpes))
        output_mold.append(output)
    max_len = max(len(x) for x in output_mold)
    mold_np = np.ones([len(sentences),max_len])
    for i in range (0,len(sentences)):
        raw_words = output_mold[i]
        len_lst.append(len(raw_words))
        mold_np[i,:len_lst[-1]]=raw_words
    seg_token = torch.LongTensor(mold_np).to(device)
    seg_token_len = torch.LongTensor(len_lst).to(device)
    return seg_token , seg_token_len

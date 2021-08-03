import mxnet as mx
from mlm.models import get_pretrained
ctxs = [mx.gpu()]
model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-cased')
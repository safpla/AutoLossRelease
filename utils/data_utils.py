import numpy as np

unknown_token = 0
start_token = 1
end_token = 2
padding_token = 3
reserved_tokens = [start_token, end_token, padding_token]

# batch preparation of a given sequence
def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths

def prepare_train_batch(inputs, targets, maxlen=None):
    '''
    inputs: a list of sentences
    targets: a list of sentences
    '''
    lengths_inputs = [len(s) for s in inputs]
    lengths_targets = [len(s) for s in targets]
    if maxlen is not None:
        new_inputs = []
        new_targets = []
        new_lengths_inputs = []
        new_lengths_targets = []
        for l_x, s_x, l_y, s_y in zip(lengths_inputs, inputs,
                                      lengths_targets, targets):
            if l_x <= maxlen and l_y <= maxlen:
                new_inputs.append(s_x)
                new_targets.append(s_y)
                new_lengths_inputs.append(l_x)
                new_lengths_targets.append(l_y)
        lengths_inputs = new_lengths_inputs
        lengths_targets = new_lengths_targets
        inputs = new_inputs
        targets = new_targets

    batch_size = len(inputs)
    out_lengths_inputs = np.array(lengths_inputs)
    out_lengths_targets = np.array(lengths_targets)

    maxlen_inputs = np.max(out_lengths_inputs)
    maxlen_targets = np.max(out_lengths_targets)

    out_inputs = np.ones((batch_size, maxlen_inputs)).astype('int32') * end_token
    out_targets = np.ones((batch_size, maxlen_targets)).astype('int32') * end_token

    for idx, [s_x, s_y] in enumerate(zip(inputs, targets)):
        out_inputs[idx, :lengths_inputs[idx]] = s_x
        out_targets[idx, :lengths_targets[idx]] = s_y

    return out_inputs, out_lengths_inputs, out_targets, out_lengths_targets

def unpaddle(predicts):
    predicts_out = []
    for predict in predicts:
        predict = [p for p in predict if p not in reserved_tokens]
        predicts_out.append(predict)
    return predicts_out


import math, random


def nonzeros(samples, maxAngleAbsDegrees=.05, verbose=True):
    '''
    Return all of the samples with an absolute turning angle
    greater than maxAngleAbsDegrees. We assume that the sample
    turning angles are specified in radians, so this value is converted.
    You may want to use this function because points with a turning angle of
    zero tend to be points when we accidentally "let off of the wheel".
    Very rarely should the angle be exactly 0.
    '''
    for i, sampleset in enumerate(samples):
        N = len(sampleset)
        window = maxAngleAbsDegrees * math.pi / 180
        samples[i] = [s for s in sampleset if math.fabs(float(s[3])) > window]
        if verbose:
            retained = len(samples[i])
            retainedPercent = 100.0 * retained / N
            removed = N - retained
            removedPercent = 100.0 * removed / N
            print('{:4d}'.format(removed),
                  "samples removed  (" + '{:0f}'.format(removedPercent) + "%)"
                  + " with driving angles of approximately 0\n"
                  + '{:4d}'.format(retained)
                  + " samples retained (" + '{:0f}'.format(retainedPercent) + "%)\n"
                  + "------------------------")
    return samples


def sort(samples):
    '''
    :param samples: A list of lists. Each sublist will be sorted based on the turning angle
    '''
    for i, sampleset in enumerate(samples):
        samples[i] = sorted(sampleset, key=lambda x: float(x[3]))
    return samples


def chunk(samples, n=10):
    '''
    :param samples: A list of lists. Each sublist will be broken into lists of n elements.
    '''
    chunked_samples = []
    for sampleset in samples:
        for i in range(0, len(sampleset), n):
            chunked_samples.append(sampleset[i:i + n])
    return chunked_samples


def shuffle(samples, seed=0):
    '''
    :param samples: A list of lists. Each sublist will be shuffled.
    :param seed: A seed to use for shuffling
    :return:
    '''
    random.seed(seed)
    for i, sampleset in enumerate(samples):
        random.shuffle(samples[i])


def split(samples, train_percent=.7):
    '''
    :param samples: A list of list. Each of these sublists will be broken up based on the
    specified training percent and then conglomerated into master training anf validation
    lists
    :param train_percent:
    :return:
    '''
    train_samples = []
    valid_samples = []
    for sampleset in samples:
        training_size = math.floor(train_percent * len(sampleset))
        train_samples.extend(sampleset[:training_size])
        valid_samples.extend(sampleset[training_size:])
    return train_samples, valid_samples

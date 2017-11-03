import csv, math, numpy, os


def getLogFolders():
    dirs = []
    offs = []
    if False:
        root = 'C:/Users/matth/Desktop/train/'
        dir = ['middle/', 'hugLeft/', 'hugRight/', 'outLeft/', 'outRight/']
        off = numpy.array([0.0, 2.0, -2.0, 5.0, -5.0]) * math.pi / 180.0
        dir = ['middle/']
        off = numpy.array([0.0]) * math.pi / 180.0
        for i, d in enumerate(dir):
            dir[i] = os.path.join(root, d)
        dirs.extend(dir)
        offs.extend(list(off))

    if False:
        root = 'C:/Users/matth/Desktop/bcloning/'
        dir = ['recovery2/']
        off = [0.0 for d in dir]
        for i, d in enumerate(dir):
            dir[i] = os.path.join(root, d)
        dirs.extend(dir)
        offs.extend(list(off))

    if False:
        root = 'C:/Users/matth/Desktop/'
        dir = ['data/', 'aug/', 'aug2/', 'aug3/', 'aug4/']
        dir = ['data/']
        off = [0.0 for d in dir]
        for i, d in enumerate(dir):
            dir[i] = os.path.join(root, d)
        dirs.extend(dir)
        offs.extend(list(off))

    if False:
        root = 'C:/Users/matth/Desktop/'
        dir = ['combined/']
        off = [0.0 for d in dir]
        for i, d in enumerate(dir):
            dir[i] = os.path.join(root, d)
        dirs.extend(dir)
        offs.extend(list(off))

    if True:
        root = 'C:/Users/matth/Desktop/sims/'
        dir = [os.path.join(root, d) for d in os.listdir(root)]
        off = [0.0 for d in dir]
        dirs.extend(dir)
        offs.extend(list(off))
    return dirs, offs


def getLogs(dir='C:/Users/matth/Desktop/bcloning/'):
    '''
    We will look for all of the driving_log.csv files in the specified directory,
    including subdirectories.
    '''
    logs = []
    logfile = 'driving_log.csv'
    for dirname, dirnames, filenames in os.walk(dir):
        if logfile in filenames:
            logs.append(os.path.join(dirname, logfile))
    return logs


def loadLog(log):
    '''
    Return all of the lines from the specified log file.
    If the file does not end with .csv, then we append 'driving_log.csv'
    '''
    if not log.endswith(".csv"):
        log = os.path.join(log, 'driving_log.csv')
    with open(log) as csvfile:
        reader = csv.reader(csvfile)
        samples = []
        for sample in reader:
            samples.append(sample)
    return samples


def loadLogs(logs=None):
    '''
    Return all of the lines from the specified log files
    '''
    if logs is None:
        logs = getLogs()
    samples = []
    for log in logs:
        samples.extend(loadLog(log))
    return numpy.array(samples)


def loadData(dirs, offsets=None):
    '''
    :param dirs: A list of directories containing driving data
    :param offsets: The offsets to use for each of the dirs. This should be the same size as dirs
    :return: A list of the samples in each directory
    '''
    allsamples = []
    if offsets is None:
        offsets = numpy.array([0.0 for dir in dirs])
    for dir, offset in zip(dirs, offsets):
        samples = loadLog(dir)
        # Add the offset
        for i, sample in enumerate(samples):
            samples[i][3] = str(float(sample[3]) + offset)
        allsamples.append(samples)
    return allsamples

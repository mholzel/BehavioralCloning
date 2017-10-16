import csv, os, sklearn
import numpy as np

def getLogs( dir = 'C:/Users/matth/Desktop/bcloning/' ):
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
    with open(log) as csvfile:
        reader = csv.reader(csvfile)
        samples = []
        for sample in reader: 
            samples.append(sample)
    return samples
    
def loadLogs(logs = None):
    if logs is None: 
        logs = getLogs()
    samples = []
    for log in logs: 
        samples.extend( loadLog(log) )
    return np.array(samples)

def sampleGenerator( samples, batch_size ):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
    
print( loadLogs().shape )
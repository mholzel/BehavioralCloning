import cv2, math, numpy, random

def sampleGenerator(samples, batch_size=128):
    N = len(samples)
    iteration = 0
    while 1:  # Loop forever so the generator never terminates
        iteration += 1
        random.seed(iteration)
        random.shuffle(samples)
        for offset in range(0, N, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, center_angle = extract(batch_sample)
                images.append(center_image)
                angles.append(center_angle)
            yield numpy.array(images), numpy.array(angles)


def sampleGeneratorWithFlipped(samples, batch_size=128):
    '''
    Note that the batch_size will actually be twice what you specify
    '''
    N = len(samples)
    iteration = 0
    while 1:  # Loop forever so the generator never terminates
        iteration += 1
        random.seed(iteration)
        random.shuffle(samples)
        for offset in range(0, N, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image, center_angle = extract(batch_sample)
                images.append(center_image)
                angles.append(center_angle)
                center_image, center_angle = extractFlipped(center_image, center_angle)
                images.append(center_image)
                angles.append(center_angle)
            yield numpy.array(images), numpy.array(angles)


def extract(batch_sample):
    center_image = cv2.imread(batch_sample[0])
    a = numpy.array(center_image).shape
    try:
        if (a[0] != 160 or a[1] != 320 or a[2] != 3):
            print(batch_sample)
    except:
        print("Dimension error. Should be (160,320,3), got : ", a,  batch_sample)
        print(a)
        print()
    center_angle = float(batch_sample[3]) * 180. / math.pi
    return center_image, center_angle


def extractFlipped(center_image, center_angle):
    return cv2.flip(center_image, 1), -center_angle

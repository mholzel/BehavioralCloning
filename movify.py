import cv2, loader, math, numpy, os, sys, tqdm
import matplotlib.pyplot as plt


def concat(images, text=None):
    """
    Concatenate a list of images together, drawing the specified text (if any) in the
    center of the concatenated image
    """
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]
    height = max(heights)
    width = sum(widths)
    cimage = numpy.zeros(shape=(height, width, 3), dtype=numpy.uint8)
    offset = 0
    for image in images:
        h, w = image.shape[:2]
        cimage[:h, offset:offset + w] = image
        offset += w
    if text is not None:
        baseline = 0
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        thickness = 2
        textSize = cv2.getTextSize(text, fontFace, fontScale, thickness)
        baseline += thickness
        x = int((width - textSize[0][0]) / 2)
        y = int(height - (height - textSize[0][1]) / 2)
        cv2.putText(cimage, text, (x, y), fontFace, fontScale, (255, 255, 255), thickness=thickness)
    return cimage


def movify(samples, output_path, fps=30):
    '''
    Create a video from all of the driving samples,
    saving the processed video at the specified output path.
    '''
    output_video = None

    # Set up the progress bar 
    progressbar = tqdm.tqdm(total=len(samples))

    # Now, for each sample, save the output in the video
    for sample in samples:

        angle = "{:2.2f}".format(float(sample[3]) * 180 / math.pi)
        images = [cv2.imread(sample[i]) for i in [1, 0, 2]]
        images[1] = concat([images[1]], text=angle)
        frame = concat(images)
        if (cv2.waitKey(10) & 0xFF == ord('q')):
            break
        if output_video is None:
            fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1')
            fourcc = 0x00000021
            output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        output_video.write(frame)
        progressbar.update()
    progressbar.close()
    cv2.destroyAllWindows()
    if output_video is not None:
        output_video.release()


if False:
    base = 'C:\\Users\\matth\\Desktop\\sims\\'
    for dir in os.listdir(base):
        if not os.path.isdir(os.path.join(base, dir)):
            continue
        print(dir)
        dir = os.path.join(base, dir) + '\\'
        samples = loader.loadData([dir])[0]
        movify(samples, os.path.join(dir, 'movie.mp4'))
else:
    dir = 'C:\\Users\\matth\\Desktop\\sims\\recovery\\'
    samples = loader.loadData([dir])[0]
    movify(samples, os.path.join(dir, 'movie.mp4'))

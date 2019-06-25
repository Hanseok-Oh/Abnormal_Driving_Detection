import av
import os
import cv2
import numpy as np

class Dataloader:
    def __init__(self, video_directory_path, resize=(64,64,3)):
        self.video_directory_path = video_directory_path
        self.video_list = self._get_video_list()
        self.resize = resize


    def choose_random_video(self):
        random_video = np.random.choice(self.video_list)
        try:
            random_video_frame = self._get_video_frame(random_video)
        except:
            random_video_frame = self._get_video_frame_cv2(random_video)

        random_video_frame = random_video_frame.astype('float32') / 255

        if self.resize:
            random_video_frame = [np.resize(i, self.resize) for i in random_video_frame]
            random_video_frame = np.array(random_video_frame)
        return random_video_frame


    def _get_video_list(self):
        video_list = os.listdir(self.video_directory_path)
        return [os.path.join(self.video_directory_path, i) for i in video_list]


    def _get_video_frame(self, video_path):
        container = av.open(video_path)
        video = container.streams.video[0]
        frames = container.decode(video=0)

        frame_list = []
        for frame in frames:
            img = frame.to_image()
            img = np.array(img)
            frame_list.append(img)
        return np.array(frame_list)


    def _get_video_frame_cv2(self, video_path):
        cap = cv2.VideoCapture(video_path) # video 불러오기
        video_frame_num = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)) # frame 수

        # capture
        frame_list = []
        for i in range(video_frame_num):
            ret, frame = cap.read()
            frame_list.append(frame)

            if cv2.waitKey(30) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return np.array(video_frame)

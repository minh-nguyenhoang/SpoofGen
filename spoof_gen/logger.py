import os
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, Queue
USE_TENSORBOARD = False
USE_COMET = False
USE_NEPTUNE = False

class DummyLogger:
    def __init__(self,*args, **kwargs) -> None:
        pass
    def add_scalar(self,*args, **kwargs):
        pass
    def add_scalars(self,*args, **kwargs):
        pass
    def add_figure(self,*args, **kwargs):
        pass
    def add_image(self,*args, **kwargs):
        pass
    def add_images(self,*args, **kwargs):
        pass
    def add_text(self,*args, **kwargs):
        pass
    def add_audio(self,*args, **kwargs):
        pass

def to_queue(tag, queue = 'queue'):
    def _wrapper(f):    
        def _wrap_func(instance,*args, **kwargs):
            getattr(instance,queue).put({"args":args,"kwargs":kwargs, "tag": tag})
            f(instance, *args, **kwargs)
        return _wrap_func
    return _wrapper


class Logger:
    def __init__(self, log_dir= None, **kwargs) -> None:
        if log_dir is None:
            import time
            log_dir = f'logs/{time.strftime ("%Y-%m-%d--%H-%M-%S")}'
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok= True)
            self.content = {}
        else:
            self.content = self.get_content()
        
        self.queue = Queue()
        self.worker = Pool(14)


    @to_queue('scalar')
    def add_scalar(self, name, value, step = -1):
        pass
    @to_queue('scalar_dict')
    def add_scalars(self, name, values, step = -1):
        pass
    def add_figure(self,name, figure):
        return
    
        path = os.path.join(self.log_dir,name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok= True)
        if path not in self.content.keys():
            self.content[path]:int = 1
            index = 1
        else:
            index = self.content[path]
            self.content[path] += 1
        if isinstance(figure, plt.Figure):
            figure.savefig(os.path.join(path, f'Figure_{index}.png'))
        else:
            cv2.imwrite(os.path.join(path, f'Figure_{index}.png'), figure)
    
    def add_text(self, name, text):
        pass
    def add_audio(self, name, audio):
        pass
    def add_asset(self, name, asset):
        pass

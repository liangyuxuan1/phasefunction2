import logging
import os
import time

# https://xnathan.com/2017/03/09/logging-output-to-screen-and-file/
# https://blog.csdn.net/weixin_43064185/article/details/97300443

class double_logger:
    '''
    logging to console and log file simultaneously
    '''
    def __init__(self, log_path=None):
        self.logger = logging.getLogger()  # 不加名称设置root logger
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 使用FileHandler输出到文件
        if log_path is None:
            log_path = os.getcwd()

        if not os.path.exists(log_path):
            os.mkdir(log_path)
            
        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
        log_file_name = os.path.join(log_path, now + '.log')
        fh = logging.FileHandler(log_file_name, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        fh.close()
        ch.close()

    def getLogger(self):
        return self.logger
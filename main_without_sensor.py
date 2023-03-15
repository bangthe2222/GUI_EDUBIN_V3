import os
import kivy
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

kivy.require('2.0.0')

from kivy.clock import Clock
from kivy.app import App

from kivy.graphics import Rectangle
from kivy.core.image import Image
from kivy.uix.image import Image as KvImage
from kivy.uix.button import Button
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import datetime
from kivy.uix.screenmanager import ScreenManager, Screen, WipeTransition
import cv2
from kivy.graphics.texture import Texture
import cv2
import playsound 
import serial
import tensorflow as tf
import numpy as np
import csv
import piexif
from kivy.core.window import Window
from kivy.uix.video import Video
from kivy.uix.videoplayer import VideoPlayer

# Window.size = (720, 405)
Window.fullscreen = 'auto'
# Window.borderless = True
Builder.load_string('''
<FancyButton>:
    background_normal: ''
    background_color: 0/255, 150/255, 70/255
    text_color: 1,0,0,1
    color: 122/255, 245/255, 51/255

''')
# DEFINE PARAMETERS

PATH_DETECT_IMAGE = ""
FOLDER_ON_SCREEN_IMAGE = "./detect_image/" 
LOCATION = "UEH_B1"
DATA_RECORD_FILE = "./images_data/data_" + LOCATION + '_' + datetime.datetime.now().strftime("%Y-%m-%d") +  ".csv"
LIST_IMAGES_FOLDER = ["./images_data/ALU/","./images_data/FOAM_BOX/" ,"./images_data/MILK_BOX/" , "./images_data/PET/", "./images_data/PLASTIC_CUP/", "./images_data/Unidentified/"]
LIST_CLASSES = ['Alu', 'Foam_box', 'Milk_box', 'PET', 'Plastic_cup', 'Undentified']

FRAME_H_SCALE = 0.4
FRAME_W_SCALE = FRAME_H_SCALE*9/16

FRAME_H_SCALE_MAIN = 0.8
FRAME_W_SCALE_MAIN = FRAME_H_SCALE_MAIN*9/16

FRAME_X_CENTER = 0.225
FRAME_Y = 0.1

ALU_COLOR = (39, 83,188)
MILKBOX_COLOR = (68,148,0)
PET_COLOR = (163,81,50)
FOAM_COLOR = (67,93,163)
UDENTIFIED_COLOR = (155,68,96)

LIST_COLOR_CLASSES = [ALU_COLOR, FOAM_COLOR, MILKBOX_COLOR, PET_COLOR, PET_COLOR, UDENTIFIED_COLOR]

def insert_exif(path_img ,author = "EduBin", camera = "desktop"):
    zeroth_ifd = {40093:author.encode("utf-16"), 33432 : author, 271 : camera, 306: datetime.datetime.now().strftime("%Y-%m-%d")}
    exif_bytes = piexif.dump({"0th": zeroth_ifd})
    piexif.insert(exif_bytes,path_img)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


class FancyButton(Button):
    opacity = 0.87

# camera config
class CameraWidget(Widget):
    def __init__(self, capture, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)

        self.capture = capture
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Clock.schedule_interval(self.update, 1.0 / 33.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        frame = cv2.flip(frame, 0)
        draw_border(frame, (0,0), (640,480), (56,104,0), 20, 30, 50)
        if ret:
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.canvas.clear()
            with self.canvas:
                Rectangle(texture=texture, pos=self.pos, size=self.size)

# main screen
class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        self.checkSensor = True
        # Tạo Widget layout
        layout = FloatLayout()

        # Tạo Widget hiển thị hình nền
        self.canvas.clear()
        with layout.canvas:
            # Load the image and create a texture from it
            img = Image('./images/screen_main.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=layout.pos, size=layout.size)

        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        layout.bind(pos=self.update_rect, size=self.update_rect)

        # Tạo các widget và thêm chúng vào màn hình
        # tạo camera capture và widget camera
        self.cap = cv2.VideoCapture(0)
        camera_widget = CameraWidget(capture=self.cap)
        camera_widget.size_hint = (FRAME_W_SCALE_MAIN, FRAME_H_SCALE_MAIN)
        camera_widget.pos_hint =  {'x': 0.285, 'y': 0.1}
        layout.add_widget(camera_widget)

        # bắt đầu cập nhật khung hình camera
        Clock.schedule_interval(camera_widget.update, 1.0 / 30.0)

        # load model
        model_name = "./vending_5class_ResNet50V2_30epoch_avg.h5"
        self.model = tf.keras.models.load_model(model_name)

        # Add FancyButton widget
        fancy_button = FancyButton(text='Chụp hình')
        fancy_button.size_hint = (None, None)
        fancy_button.size = (200, 50)
        fancy_button.pos_hint = {'center_x': 0.8, 'y': 0.25}
        fancy_button.bind(on_press=self.detectImage)
        layout.add_widget(fancy_button)
        self.add_widget(layout)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def detectImage(self, *args):
        # time.sleep(0.5)

        # threshold accept
        threshold_accept = 0.45

        # Đọc khung hình từ camera
        ret, frame = self.cap.read()

        # resize image
        image_src = cv2.resize(frame.copy(),(224,224))
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        image = np.asarray([image_src])

        # predict image
        predict = self.model.predict(image)
        # predict = [[0.5, 0.4,0.1]]
        id_out = np.argmax(predict[0])

        # get accuracy
        acc_pre = predict[0][id_out]

        print(predict[0][id_out])

        # check accuracy threshold
        if acc_pre < threshold_accept:
            id_out = len(LIST_CLASSES) - 1
        print(id_out)
        # get date time now
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.datetime.now().strftime("%H-%M-%S")
        # image file name
        filename = LIST_IMAGES_FOLDER[id_out] + "IMAGE_" + LIST_CLASSES[id_out]+ '_' + LOCATION + '_' + date_str + "_" + time_str + ".jpg"
        
        # record data
        data = [filename, LIST_CLASSES[id_out], acc_pre,  date_str, time_str.replace('-',':') , LOCATION]
        
        # write data to record file
        with open(DATA_RECORD_FILE, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(data)
            f.close()
        

        # get global path name send to order class
        for file in os.listdir(FOLDER_ON_SCREEN_IMAGE):
            print(FOLDER_ON_SCREEN_IMAGE + file)
            os.remove(FOLDER_ON_SCREEN_IMAGE + file)

        global PATH_DETECT_IMAGE       
        PATH_DETECT_IMAGE = FOLDER_ON_SCREEN_IMAGE + "detect_" + time_str + ".png"
        # Lưu khung hình thành file ảnh
        cv2.imwrite(filename, frame)
        draw_border(frame, (0,0), (640,480), LIST_COLOR_CLASSES[id_out], 20, 30, 50)
        cv2.imwrite(PATH_DETECT_IMAGE, frame)
        insert_exif(filename)

        # check threshold and change screen
       
        time_delay = 4
        if acc_pre > threshold_accept:
            ketQua = id_out

            # ALU screen
            if ketQua == 0:
                self.manager.current = 'screen_ALU'
                Clock.schedule_once(self.play_sound_ALU,1)
            elif ketQua == 1:
                self.manager.current = 'screen_FOAM'
                Clock.schedule_once(self.play_sound_FOAM,1)
                time_delay = 5
            # PET screen
            elif ketQua == 3 or ketQua == 4:
                self.manager.current = 'screen_PET'
                Clock.schedule_once(self.play_sound_PET,1)

            # MILKBOX screen
            elif ketQua == 2:
                self.manager.current = 'screen_MILKBOX'
                Clock.schedule_once(self.play_sound_MILKBOX,1)
                time_delay = 3
        else:
            # unknow object screen
            self.manager.current = 'screen_Unidentified'
            Clock.schedule_once(self.play_sound_Undentifided,1)
            time_delay = 3
        # wait 4s and change to main screen
        Clock.schedule_once(self.reset_camera, time_delay)
        
    def reset_camera(self, *args):
        # Quay lại màn hình camera
        self.manager.current = 'camera_screen'
        self.checkSensor = True

    # play sound 
    def play_sound_ALU(self, path):
        playsound.playsound("./voice/intro.mp3")
        playsound.playsound("./voice/voice_ALU.mp3")

    def play_sound_PET(self, path):
        playsound.playsound("./voice/intro.mp3")
        playsound.playsound("./voice/voice_PET.mp3")
        
    def play_sound_MILKBOX(self, path):
        playsound.playsound("./voice/intro.mp3")
        playsound.playsound("./voice/voice_MILKBOX.mp3")

    def play_sound_FOAM(self, path):
        playsound.playsound("./voice/intro.mp3")
        playsound.playsound("./voice/voice_FOAM.mp3")

    def play_sound_Undentifided(self, path):
        playsound.playsound("./voice/intro.mp3")
        playsound.playsound("./voice/voice_undentified.mp3")

# ALU screen
class screenDetect(Screen):
    def __init__(self, path_image, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.canvas.clear()
        with self.canvas:
            # Load the image and create a texture from it
            img = Image(path_image).texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object
        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)
        Clock.schedule_interval(self.update_detect_img, 0.5)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_detect_img(self, dt):
        global PATH_DETECT_IMAGE
        self.clear_widgets()
        self.small_img = KvImage(source=PATH_DETECT_IMAGE, size_hint=(FRAME_W_SCALE, FRAME_H_SCALE), allow_stretch = True, keep_ratio = False)
        self.small_img.pos_hint = {'center_x': FRAME_X_CENTER, 'y': FRAME_Y}
        self.add_widget(self.small_img)

# intro sreen
class MyScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.canvas.clear()
        with self.canvas:
            # Load the image and create a texture from it
            img = Image('./images/screen_intro.png').texture
            # Create a Rectangle object with the texture as its source
            self.rect = Rectangle(texture=img, pos=self.pos, size=self.size)
        # Bind the texture and size properties of the Rectangle object

        # to the corresponding properties of the widget
        self.bind(pos=self.update_rect, size=self.update_rect)

        fancy_button = FancyButton(text='Bắt đầu')
        fancy_button.size_hint = (None, None)  # Chỉ định kích thước cố định cho nút
        fancy_button.size = (200, 50)  # Thiết lập kích thước cho nút
        fancy_button.pos_hint = {'center_x': 0.5, 'y': 0.1, }  # Đặt nút ở giữa dưới khung hình
        fancy_button.bind(on_press=self.on_press)
        self.add_widget(fancy_button)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def on_press(self, *args):
        # tạm dừng thực thi chương trình trong 1 giây
        Clock.schedule_once(self.change_screen, 1)

    def change_screen(self, *args):
        # kiểm tra xem self.manager có khác None hay không
        if self.manager is not None:
            self.manager.current = 'ads_screen'

class ADS_Screen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        video = Video(source='./videoads.mp4')
        video.state='play'
        video.options = {'eos': 'loop'}
        video.allow_stretch=True
        self.add_widget(video)
        
 

# init app 
class MyApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create detect image folder

        if not os.path.exists(FOLDER_ON_SCREEN_IMAGE):
            os.makedirs(FOLDER_ON_SCREEN_IMAGE)

        # create images data folder
        
        for path_image in LIST_IMAGES_FOLDER:
            if not os.path.exists(path_image):
                os.makedirs(path_image)
        
        # create data record file
       
        if not os.path.exists(DATA_RECORD_FILE):
            header = ['path', 'class', 'accuracy','date', 'time','location']
            with open(DATA_RECORD_FILE, 'w+', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the header
                writer.writerow(header)
                f.close()

    def build(self):
        self.title = 'EduBin'
        sm = ScreenManager(transition=WipeTransition())
        sm.add_widget(MyScreen(name='my_screen'))
        sm.add_widget(CameraScreen(name='camera_screen'))
        sm.add_widget(ADS_Screen(name='ads_screen'))
        sm.add_widget(screenDetect(path_image="./images/screen_MILKBOX.png",name='screen_MILKBOX'))
        sm.add_widget(screenDetect(path_image="./images/screen_ALU.png",name='screen_ALU'))
        sm.add_widget(screenDetect(path_image="./images/screen_PET.png",name='screen_PET'))
        sm.add_widget(screenDetect(path_image="./images/screen_FOAM.png",name='screen_FOAM'))
        sm.add_widget(screenDetect(path_image="./images/screen_Undentified.png",name='screen_Unidentified'))
        return sm


if __name__ == '__main__':
    MyApp().run()

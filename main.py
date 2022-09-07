import numpy as np
import onnx
import onnxruntime
import PIL.Image

# for OpenCV
from pickle import FALSE
import cv2 
from time import time 

PROB_THRESHOLD = 0.2  # 値以上の確率のもののみ検知する


# エクスポートしたモデルの処理
# 参考：https://github.com/Azure-Samples/customvision-export-samples/tree/main/samples/python/onnx
class Model:
    def __init__(self, model_filepath):
        self.session = onnxruntime.InferenceSession(str(model_filepath))
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filepath)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True

    def predict(self, image_filepath):
        image = PIL.Image.open(image_filepath).resize(self.input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

    
###############################################
# OpenCVの設定
###############################################
CAMERA_NUM = 0   # 0:内蔵カメラ 1:USBカメラ

# 読み込む対象のカメラの指定
cam = cv2.VideoCapture(CAMERA_NUM, cv2.CAP_DSHOW)
# Windowサイズの指定（横）
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# Windowサイズの指定（縦）
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# カメラ映像からのスクリーンショットを切り出すタイミングを指定(秒)
limit_time = 0.1
# 前回読み込んだ時間の初期化
previous_time = 0

###############################################
# メイン処理
###############################################
# エクスポートしたモデルの読み込み
model = Model("model/model.onnx")
    
while True:
    try:
        # カメラ映像からスクリーンショットの取得
        _, img = cam.read()
        # 指定間隔を超えたら実行
        if limit_time < time() - previous_time:
            # スクリーンショットの取得
            cv2.imwrite('./tmp/screen.jpg', img)
            # エクスポートしたモデルで推論、以下形式でレスポンスが返ってくる
            # Label: 0, Probability: 0.05247, box: (0.46848, 0.93321) (0.49320, 0.97316)
            outputs = model.predict("./tmp/screen.jpg")
            # 前回取得時間を更新
            previous_time = time()

        text_color = (255, 255, 255)  # 白で初期化
        
        height, width = img.shape[:2]
        label = {0: "dent", 1:"scratch", 2:"red_ink", 3:"black_ink"}
        for box, class_id, score in zip(outputs['detected_boxes'][0], outputs['detected_classes'][0], outputs['detected_scores'][0]):
            if score > PROB_THRESHOLD:
                #print(f"Label: {class_id}, Probability: {score:.5f}, box: ({box[0]:.5f}, {box[1]:.5f}) ({box[2]:.5f}, {box[3]:.5f})")
                
                #　不良種別に色分け (Blue, Green, Red)
                if class_id == 0: #凹み
                    text_color = (255, 255, 255) #白
                elif class_id == 1: #傷
                    text_color = (255, 0, 0) #青
                elif class_id == 2: #赤インク
                    text_color = (0, 0, 255) #赤
                elif class_id == 3: #黒インク
                    text_color = (0, 0, 0) #黒
                
                # 不良種別の描画
                img = cv2.putText(img,
                text=str(label[class_id]) + " " + str(round(score,2)),
                org=(int(box[0]*width)-30, int(box[1]*height)-10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=text_color,
                thickness=2,
                lineType=cv2.LINE_4)
                
                # BBoxの描画
                img = cv2.rectangle(img,
                        (int(box[0]*width), int(box[1]*height)),
                        (int(box[2]*width), int(box[3]*height)),
                        text_color,
                        2)
        # リアルタイム検知画面の表示
        cv2.imshow("detect", img)

        # QかEscキー押下にてプログラムを終了できる設定
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        
    # エラーが発生した場合の処理
    except Exception as e:
        import traceback
        print("[ERROR] ", traceback.print_exc())
        continue

# 終了キー押下後、後処理
cam.release()
cv2.destroyAllWindows()
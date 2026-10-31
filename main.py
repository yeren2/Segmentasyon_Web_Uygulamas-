import cv2
import ultralytics
import numpy as np
from torch.fx.experimental.unification.multipledispatch.dispatcher import source
from ultralytics import YOLO
from flask import Flask,render_template,request,jsonify,send_file
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='templates')

upload_folder="uploads"
allowed_extension={"png","jpg","jpeg","gif","bmp"}
max_file_size=10*1024*1024

os.makedirs(upload_folder,exist_ok=True)
os.makedirs("static",exist_ok=True)

app.config["UPLOAD_FOLDER"]=upload_folder
app.config["MAX_CONTENT_LENGTH"]=max_file_size

model_path="yolo26n-seg.pt"
model=YOLO(model_path)

np.random.seed(42)
colors=np.random.randint(0,255,(80,3),dtype="uint8")

#sisteme yüklenen dosyanın güvenli veya izin verilen formatta olup olmadığını kontrol eder
def allowed_file(filename):# return '.' in filename= burada filename içinde nokta olup olmadığını kontrol eder(uzantı olması için nokta lazım çünkü)

    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in allowed_extension
        #filename.rsplit('.',1)=sağdan başlayarak noktaya göre böler,ikinci parametre [1] işlemin 1 kere yapılacağını belirtir
        #Örnek: Dosya adı fotograf.yedek.png ise, sağdaki ilk noktadan böler ve geriye iki elemanlı bir liste döndürür: ['fotograf.yedek', 'png'].
def extract_data(img,model):
    h,w,ch=img.shape
    results=model.predict(source=img.copy(),save=False,save_txt=False)#predict görselin kopyasını modele göndererek nesne tespiti ve segmentasyon yapar
    result=results[0]
    if result.masks is None:
        return [],[],[],[],{}
    class_names=result.names
    seg_contour_idx=[]
    for seg in result.masks.xyn:
        seg[:,0]=seg[:,0]*w
        seg[:,1]=seg[:,1]*h
        segment=np.array(seg,dtype=np.int32)
        seg_contour_idx.append(segment)
    bbox=np.array(result.boxes.xyxy.cpu(),dtype="int")
    class_id=np.array(result.boxes.cls.cpu(),dtype="int")
    scores=np.array(result.boxes.conf.cpu(),dtype="float").round(2)
    return bbox,class_id,seg_contour_idx,scores,class_names
def process_img(img):
    font=cv2.FONT_HERSHEY_SIMPLEX
    boxes,class_id,seg_contour_idx,scores,class_names=extract_data(img,model)

    if len(boxes)==0:
        return img,[]
    overlay=img.copy()
    detect_object=[]

    for box,class_id,seg_contour,score in zip(boxes,class_id,seg_contour_idx,scores):
        xmin,ymin,xmax,ymax=box
        color=[int(c) for c in colors[class_id]]
        class_name=class_names[class_id]
        label=f"{class_name}:{score}"

        cv2.fillPoly(overlay,[seg_contour],color)#görseli,modelin bulunduğu nesnenin içinde tamamen o renge boyar
        cv2.polylines(img,[seg_contour],True,color,2)

        (label_width,label_height),baseline=cv2.getTextSize(label,font,0.5,1)
        label_ymin=max(ymin,label_height+1)
        cv2.rectangle(img,(xmin,label_ymin-label_height-10),
                      (xmin+label_width,label_ymin+baseline-10),color,cv2.FILLED)

        cv2.putText(img,label,(xmin,label_ymin-5),font,0.5,(255,255,255),1)
        detect_object.append({
            "class":class_name,
            "confidence":float(score),
            "bbox":[int(xmin),int(ymin),int(xmax),int(ymax)],
        })
    alpha=0.3
    img=cv2.addWeighted(overlay,alpha,img,1-alpha,0)
    return img,detect_object

def image_to_base64(img):#opencv formatındaki görseli internet tarayıcısında HTML veya JSON formatında doğrudan görüntüleyebilecek base64 formatında döndürür
    _,buffer=cv2.imencode(".png",img)#_ başarıyla okundu mu onu kontrol eder,buffer sıkıştırılmış görüntü verisidir,imencode=> belirtilen formata göre sıkıştırır ve kodlar
    img_base64=base64.b64encode(buffer).decode("utf-8")#encode bineryi base64 e çevirir,decode ise bite ları stringe dönüştürür,jsona veya html e gönderebilmek için
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():#tarayıcıya sitenin adresi yazılınca anasayfa çıkar
    """Ana Sayfa"""
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload_file():#/upload adresine sadece veri gönderimi(POST)yapılabilir
    try:
        if "file" not in request.files:
            return jsonify({"error":"dosya bulunamadı"}),400
        file=request.files["file"]

        if file.filename=="":
            return jsonify({"error":"Dosya seçilmedi"}),400
        if not allowed_file(file.filename):
            return jsonify({"error":"Geçersiz dosya formatı"}),400
        filename=secure_filename(file.filename)#dosya adındakı tehlikeli karakterleri temizler
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],filename)#Sunucuda dosyaların yükleneceği klasörün yolu ile dosya adını birleştirir (Örn: uploads/resim.jpg).
        file.save(filepath)

        img=cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({"error":"Görüntü okunamadı"}),400

        original_base64=image_to_base64(img)
        processed_img, detect_objects = process_img(img)
        processed_base64=image_to_base64(processed_img)

        os.remove(filepath)#gereksiz yer kaplamasın diye dosyayı siliyoruz
        return jsonify({
            "original_image": original_base64,
            "processed_image": processed_base64,
            "objects": detect_objects
        }), 200
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/health')#yapay zeka modelinin yüklenip yüklenmediğini kontrol eder
def health_check():
    return jsonify({"status":"healthy","model_loaded":model is not None})#yapay zeka modelinin yüklenip yüklenmediğini kontrol eder

if __name__=="__main__":#sunucuyu başlatma bloğu
    app.run(debug=False,host="0.0.0.0",port=5000,use_reloader=False)#• Flask'ın dahili web sunucusunu ayağa kaldırır ve uygulamayı dinlemeye başlar.
    #debug=False: Hata ayıklama (debug) modunu kapatır. Canlıya alınan projelerde güvenlik ve performans için bu mod her zaman kapatılmalıdır.
#host="0.0.0.0": Sunucunun sadece kendi içinde (127.0.0.1) değil, aynı ağdaki diğer tüm cihazlardan ve dış dünyadan gelen isteklere de açık olacağını belirtir. Projeyi bir sunucuya veya Docker'a koyduğunuzda bu ayar zorunludur.
#use_reloader=False: Kodda bir değişiklik yaptığınızda Flask'ın sunucuyu otomatik olarak yeniden başlatmasını engeller. Özellikle arka planda büyük yapay zeka modelleri (YOLO vb.) yüklenirken bu ayarın False olması çok önemlidir


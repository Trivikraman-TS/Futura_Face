import os
import gdown

os.makedirs('pretrained_models', exist_ok=True)

# Download pretrained models
gdown.download("https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download", "pretrained_models/sam_ffhq_aging.pt", quiet=False)
gdown.download("https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat", "pretrained_models/shape_predictor_68_face_landmarks.dat", quiet=False)
gdown.download("https://drive.google.com/uc?id=1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0", "pretrained_models/psp_ffhq_encode.pt", quiet=False)
gdown.download("https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT", "pretrained_models/stylegan_ffhq.pt", quiet=False)
gdown.download("https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn", "pretrained_models/IR_SE50_Model.pt", quiet=False)
gdown.download("https://drive.google.com/uc?id=1atzjZm_dJrCmFWCqWlyspSpr3nI6Evsh", "pretrained_models/vgg_age_classifier.pt", quiet=False)
import cv2
import os
from IPython import embed

out = "out"
os.makedirs(out, exist_ok=True)

image_root_path = "/home/dhiraj/project/CFBI/robot_data/rgb/test"
label_path = "result/resnet101_cfbi/eval/test/test_resnet101_cfbi_ckpt_unknown/Annotations"
folders = os.listdir(label_path)

for f in folders:
    img_seq = sorted(os.listdir(os.path.join(label_path, f)))
    out_f = os.path.join(out, f)
    os.makedirs(out_f, exist_ok=True)
    for i in img_seq:
        img = cv2.imread(os.path.join(image_root_path, f, i.split(".")[0] + ".jpg"))
        img = cv2.resize(img, (400, 400))
        label = cv2.imread(os.path.join(label_path, f, i))

        # # TODO: remove
        # label *= 0
        # label[200:] += 1
        temp = img[label[:, :, 2] > 0]
        temp[:, 2] = 255
        img[label[:, :, 2] > 0] = temp
        cv2.imwrite(os.path.join(out_f, "{}.jpg".format(i)), img)

    # run the fi maker on folder images
    os.system("convert -delay 20 -loop 0 {}/*.jpg {}/{}.gif".format(out_f, out, f))
    os.system("rm -rf {}".format(out_f))
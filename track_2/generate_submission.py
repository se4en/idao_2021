import torch
import csv
import os
import logging
from torchvision.transforms.functional import to_tensor

from model import SimpleConv
from preparation import prep_img


def submit(class_model, reg_model,threshold=0.5, round_on=True):
    with open('submission.csv' , 'wt') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id','classification_predictions','regression_predictions'])

        public_path = "tests/public_test/"
        private_path = "tests/private_test/"

        public_test = os.listdir(public_path)
        private_test = os.listdir(private_path)
        # public
        for name in public_test:
            print("1")
            img = prep_img(public_path + name)
            print("2")
            res_img = torch.reshape(to_tensor(img), (1, 1, img.shape[0], img.shape[1]))
            #res_img = res_img.cuda()
            # classification
            class_res = int(class_model.forward(res_img)>threshold)
            # regression
            reg_res = reg_model.forward(res_img)
            if round_on:
                if reg_res <= 1.5:
                    reg_res = 1.0
                elif reg_res <= 4.5:
                    reg_res = 3.0
                elif reg_res <= 8.0:
                    reg_res = 6.0
                elif reg_res <= 15.0:
                    reg_res = 10.0
                elif reg_res <= 25.0:
                    reg_res = 20.0
                else:
                    reg_res = 30.0
            else:
                if reg_res < 1.0:
                    reg_res = 1.0
                elif reg_res > 30.0:
                    reg_res = 30.0
                else:
                    reg_res = float(reg_res)
            # write
            writer.writerow([name[:-4], class_res, reg_res])
        # private
        for name in private_test:
            img = prep_img(private_path + name)
            res_img = torch.reshape(to_tensor(img), (1, 1, img.shape[0], img.shape[1]))
            #res_img = res_img.cuda()
            # classification
            class_res = int(class_model.forward(res_img)>threshold)
            # regression
            reg_res = reg_model.forward(res_img)
            if round_on:
                if reg_res <= 1.5:
                    reg_res = 1.0
                elif reg_res <= 4.5:
                    reg_res = 3.0
                elif reg_res <= 8.0:
                    reg_res = 6.0
                elif reg_res <= 15.0:
                    reg_res = 10.0
                elif reg_res <= 25.0:
                    reg_res = 20.0
                else:
                    reg_res = 30.0
            else:
                if reg_res < 1.0:
                    reg_res = 1.0
                elif reg_res > 30.0:
                    reg_res = 30.0
                else:
                    reg_res = float(reg_res)
            # write
            writer.writerow([name[:-4], class_res, reg_res])


if __name__ == "__main__":
    logging.info("Loading class_model")
    class_best = SimpleConv("classification")
    class_best.load_state_dict(torch.load("best_class.pickle", map_location=torch.device('cpu')))

    reg_best = SimpleConv("regression")
    logging.info("Loading reg_model")
    reg_best.load_state_dict(torch.load("best_reg.pickle", map_location=torch.device('cpu')))

    logging.info("Start submit")
    submit(class_best, reg_best)

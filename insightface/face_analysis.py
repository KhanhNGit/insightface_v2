from __future__ import division

import os
import os.path as osp
from pathlib import Path
from .retinaface import RetinaFace
# from .common import Face
from .arcface import ArcFaceONNX
from .face_align import norm_crop

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, root=osp.join(Path(__file__).parent.parent.absolute(), 'buffalo_sc')):
        self.models = {}
        model_name = os.listdir(root)
        for mod in model_name:
            if mod == 'det_500m.onnx':
                mods = os.path.join(root, mod)
                self.models['detection'] = RetinaFace(mods)
            if mod == 'w600k_mbf.onnx':
                mods = os.path.join(root, mod)
                self.models['recognition'] = ArcFaceONNX(mods)

        self.det_model = self.models['detection']
        self.rec_model = self.models['recognition']


    def prepare(self, ctx_id=0, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        # print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)


    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        # ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     face = Face(bbox=bbox, kps=kps, det_score=det_score)
        #     for taskname, model in self.models.items():
        #         if taskname=='detection':
        #             continue
        #         model.get(img, face)
        #     ret.append(face)
        # return ret

        faces = [norm_crop(img, landmark=x) for x in kpss]
        return self.rec_model.get(faces)

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            # if face.gender is not None and face.age is not None:
            #     cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg
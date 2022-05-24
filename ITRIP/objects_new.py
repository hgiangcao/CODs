import glob2
from pathlib import Path
from functools import reduce
from operator import add
import numpy as np

from pyrep.const import TextureMappingMode
from pyrep.objects import Shape, Dummy
from pyrep import PyRep
from pathlib import Path


from ITRIP.assets import GRASPNET_TTM_ROOT

EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']

graspnet_splits = {'demo':[0,2, 5, 7],'train': [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 27, 29, 30, 34, 36, 37, 38, 40, 41, 43, 44, 46, 48, 51, 52, 56, 57, 58, 60, 61, 62, 63, 66, 69, 70], 'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 38, 39, 41, 42, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77,
                                                                                                                                                                                                 78, 79, 80, 81, 82, 83, 84, 85, 86, 87], 'test_novel': [3, 4, 6, 17, 19, 20, 24, 26, 27, 28, 30, 31, 32, 33, 35, 45, 47, 49, 51, 55, 62, 63, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87], 'test_seen': [0, 2, 5, 7, 8, 9, 11, 14, 17, 18, 20, 21, 22, 26, 27, 29, 30, 38, 41, 48, 51, 52, 58, 60, 61, 62, 63, 66], 'test_similar': [1, 3, 4, 6, 10, 12, 13, 16, 19, 23, 25, 35, 39, 42, 50, 53, 54, 59, 64, 65, 67, 68]}
graspnet_obj_to_exclude = [9, 18, 19, 20, 26, 27, 28, 29, 30, 31, 33, 34, 43, 48, 50, 53, 54, 55, 56, 67, 70, 72, 74, 75, 77, 78, 79, 80, 81, 83, 85, 86]


graspnet_train = graspnet_splits["train"]
graspnet_train = set(graspnet_train) - set(graspnet_obj_to_exclude)
graspnet_train = list(graspnet_train)

graspnet_test = graspnet_splits["test_novel"]
graspnet_test = set(graspnet_test) - set(graspnet_obj_to_exclude)
graspnet_test = list(graspnet_test)

def load_graspnet_models(pr: PyRep, use_pbar=True, split=None):
    if split is None:
        graspnet_ttms = list(Path(GRASPNET_TTM_ROOT).glob('*.ttm'))
        print (graspnet_ttms)
    else:
        print ("Using ",split," set  from Graspnet ")
        ttm_nums = graspnet_splits[split]
        #ttm_nums = set(ttm_nums) - set(graspnet_obj_to_exclude)
        ttm_nums = list(ttm_nums)
        graspnet_ttms = [ Path(GRASPNET_TTM_ROOT).joinpath(str(num).zfill(3)+".ttm" )for num in ttm_nums]


    grasp_net_root_dummy = Dummy.create()
    grasp_net_root_dummy.set_name('GraspNet_objects_dummy')
    grasp_net_root_dummy.set_model(True)

    def pbar(x): return x
    if use_pbar:
        from tqdm import tqdm
        pbar = tqdm

    for ttm in pbar(graspnet_ttms):
        model = pr.import_model(str(ttm))
        model.set_parent(grasp_net_root_dummy)

    grasp_net_root_dummy.set_position([0, 0, -10])

    print("GraspNet models loaded")
    print (ttm_nums)


class TextureRandomizer:
    def __init__(self, pr: PyRep, root_path: str) -> None:
        self.pr = pr
        self.images_path = reduce(add, [
            glob2.glob(str(Path(root_path).joinpath(F"**/*.{ext}")))
            for ext in EXTENSIONS
        ])

        #self.images_path = [glob2.glob(str(Path(root_path).glob(F"*.{ext}"))) for ext in EXTENSIONS]


        print ("loaded",len(self.images_path),"texture")
        print (self.images_path)

        assert len(self.images_path
                   ) > 0, "Did not find images for texture randomization."

        try:
            texture_root = Dummy('texture_dummy')
            textures_object = texture_root.get_objects_in_tree()
            self.textures = []

            for t in textures_object:
                self.textures.append(t.get_texture())


            self.nTexture = len(self.textures)
            # print (self.nBackground," backgrounds")
            # print(self.nTexture, " texture")
        except:
            pass

    def randomize_old(self,
                  obj: Shape,
                  mapping_mode=TextureMappingMode.CUBE) -> None:
        """
        Assumes obj has a childen with visible in its name.
        If not, change the texture of the object passed in.
        """
        img_path = np.random.choice(self.images_path)
        img_path = np.random.choice(self.images_path)

        # get first children with visible in its name
        obj2texture = obj
        for c in obj.get_objects_in_tree():
            if not isinstance(c, Shape):
                continue

            if 'visible' in c.get_name():
                obj2texture = c
                break

        # create texture
        print(img_path)
        shape, tex = self.pr.create_texture(img_path)
        obj2texture.set_texture(tex, mapping_mode)
        shape.remove()

        del shape, tex

    def randomize(self,
                  obj: Shape,
                  mapping_mode=TextureMappingMode.CUBE) -> None:

        obj2texture = obj
        for c in obj.get_objects_in_tree():
            if not isinstance(c, Shape):
                continue

            if 'visible' in c.get_name():
                obj2texture = c
                break

        texture,mode = self.getRandomTexture()
        obj2texture.set_texture(texture,mode)

    def getRandomTexture(self):
        randomID = np.random.choice(np.arange(self.nTexture))
        mode = np.random.choice([TextureMappingMode.PLANE, TextureMappingMode.CYLINDER, TextureMappingMode.SPHERE,
                                 TextureMappingMode.CUBE])
        return self.textures[randomID], mode


class ObjectLoader:
    def __init__(self,
                 exclude=[],
                 load_thirteen=True,
                 load_lab_obj=False,
                 load_ycb=False,
                 load_graspnet=False) -> None:
        self.exclude = exclude
        self.objects = {}

        if load_thirteen:
            self.objects.update(self._get_obj_dic(
                'Thirteen_objects_dummy', 'thirteen'))
        elif not load_lab_obj and not load_graspnet:
            Dummy('Thirteen_objects_dummy').remove()

        if load_ycb:
            self.objects.update(self._get_obj_dic('YCB_objects_dummy', 'ycb'))
        if load_graspnet:
            self.objects.update(self._get_obj_dic('GraspNet_objects_dummy', 'graspnet'))
        if load_lab_obj:
            self.objects.update(self._get_obj_dic('Lab_objects_dummy', 'lab_obj'))

    def _get_obj_dic(self, root_dummy_name, dataset_name):
        obj_root = Dummy(root_dummy_name)

        objects = obj_root.get_objects_in_tree()
        obj_dic = {obj.get_name().lower(): obj for obj in objects}

        #print (obj_dic)
        print (objects)

        objects = {}
        print (obj_dic)

        for k, v in obj_dic.items():
            if '_visible' in k or '_skip' in k:
                continue

            col = v
            vis = obj_dic[k + '_visible']

            col.set_name(F"{dataset_name}_{col.get_name()}_")

            objects[k] = (col, vis)

        key_list = list(objects.keys())
        print (key_list)
        print ("loaded", len(objects), "objects")
        return objects

    def __len__(self):
        return len(self.objects)

    def get_random_obj(self, exclude=None,objectID = None):
        if exclude is None:
            exclude = self.exclude

        if (objectID is None):
            left_names = set(self.objects.keys()) - set(exclude)
            left_names = list(left_names)
            rand_key = np.random.choice(left_names)
        else:
            rand_key = str(objectID).zfill(3)

        o_col, _ = self.objects[rand_key]

        col = o_col.copy()

        for child in o_col.get_objects_in_tree():
            c = child.copy()
            c.set_parent(col)

        col.set_dynamic(True)
        col.set_respondable(True)

        return col,rand_key



'''
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.const import PrimitiveShape, TextureMappingMode
from pyrep.objects import Shape
from pyrep.textures.texture import  Texture

class ObjectLoader():

    def __init__(self, exclude=[]) -> None:

        obj_root = Dummy('Thirteen_objects_dummy')

        objects = obj_root.get_objects_in_tree()
        obj_dic = {obj.get_name().lower(): obj for obj in objects}
        try :
            texture_root = Dummy('texture_dummy')
            background_root = Dummy('background_dummy')
            textures_object = texture_root.get_objects_in_tree()
            textures_background = background_root.get_objects_in_tree()
            self.textures  = []
            self.backgrounds = []

            for t in textures_object:
                self.textures.append(t.get_texture())

            for t in textures_background:
                self.backgrounds.append(t.get_texture())

            self.nTexture = len(self.textures)
            self.nBackground = len(self.backgrounds)

            #print (self.nBackground," backgrounds")
            #print(self.nTexture, " texture")
        except:
            pass

        self.objects = []

        for k, v in obj_dic.items():

            if 'visible' in k:
                continue

            if k in exclude:
                continue

            col = v
            vis = obj_dic[k + '_visible']

            self.objects.append((col, vis))



    def __len__(self):
        return len(self.objects)

    def get_random_obj(self, idx=None, randomTexture=True):

        if idx is None:
            idx = np.random.choice(np.arange(len(self.objects)))
            
        col, vis = self.objects[idx]
        
        col = col.copy()
        vis = vis.copy()

        col.setVisual(vis)
        if (randomTexture):
            randomID  = np.random.choice(np.arange(self.nTexture))
            col.changeTexture(self.textures[randomID],TextureMappingMode.CYLINDER)

        vis.set_parent(col)
        col.set_dynamic(True)
        col.set_respondable(True)
        
        return col

    def getRandomTexture(self):
        randomID = np.random.choice(np.arange(self.nTexture))
        mode = np.random.choice([TextureMappingMode.PLANE,TextureMappingMode.CYLINDER,TextureMappingMode.SPHERE,TextureMappingMode.CUBE])
        return self.textures[randomID],mode

    def getRandomBackground(self):
        randomID = np.random.choice(np.arange(self.nBackground))
        mode = TextureMappingMode.PLANE
        print (randomID)
        return self.backgrounds[randomID],mode


'''
from typing import List

from pathlib import Path
from functools import reduce
from operator import add
import numpy as np

from pyrep.const import TextureMappingMode, PrimitiveShape
from pyrep.objects import Shape, Dummy
from pyrep import PyRep
import random
EXTENSIONS = ['png', 'jpg', 'jpeg', 'webp']


class TextureRandomizer:
    def __init__(self, pr: PyRep, root_path: str) -> None:
        self.pr = pr
        '''
        self.images_path = reduce(add, [
            glob2.glob(str(Path(root_path).joinpath(F"**/*.{ext}")))
            for ext in EXTENSIONS
        ])
        '''

        assert len(self.images_path
                   ) > 0, "Did not find images for texture randomization."

    def randomize(self,
                  obj: Shape,
                  mapping_mode=TextureMappingMode.CUBE) -> None:
        """
        Assumes obj has a childen with visible in its name.
        If not, change the texture of the object passed in.
        """

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
        shape, tex = self.pr.create_texture(img_path)
        obj2texture.set_texture(tex, mapping_mode)
        shape.remove()

        del shape, tex


class ObjectLoader:
    def __init__(self, exclude=[],nObject = 10) -> None:
        obj_root = Dummy('Thirteen_objects_dummy')

        objects = obj_root.get_objects_in_tree()
        obj_dic = {obj.get_name().lower(): obj for obj in objects}

        self.exclude = exclude
        self.objects = {}

        for k, v in obj_dic.items():
            if 'visible' in k:
                continue

            col = v
            vis = obj_dic[k + '_visible']

            self.objects[k] = (col, vis)

        self.baseCylinder = Shape.create(type = PrimitiveShape.CYLINDER, size = [0.08,0.08,0.08])

    def __len__(self):
        return len(self.objects)

    def get_random_obj(self, exclude=None):
        if exclude is None:
            exclude = self.exclude

        left_names = set(self.objects.keys()) - set(exclude)
        left_names = list(left_names)
        rand_key = np.random.choice(left_names)

        col, vis = self.objects[rand_key]

        col = col.copy()
        vis = vis.copy()

        vis.set_parent(col)
        col.set_dynamic(True)
        col.set_respondable(True)

        return col

    def getCylider(self):
        obj = self.baseCylinder.copy()
        obj.set_dynamic(True)
        obj.set_respondable(True)
        r, g, b = random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0, random.randint(0, 255) / 255.0
        obj.set_color([r, g, b])
        return obj


# def load_obj(paths, base: Dummy) -> Shape:
#     # col_path = paths['col']
#     vis_path = paths['vis']

#     resp = Shape.import_mesh(str(vis_path))
#     # vis = Shape.import_mesh(str(vis_path))
#     # resp.set_renderable(False)
#     # vis.set_renderable(True)
#     # vis.set_parent(resp)
#     # vis.set_dynamic(False)
#     # vis.set_respondable(False)
#     # resp.set_dynamic(True)
#     # resp.set_mass(0.5)
#     # resp.set_respondable(True)
#     # resp.set_model(True)
#     # resp.set_parent(base)
#     # resp.set_bullet_friction(1)

#     resp.set_renderable(True)
#     resp.set_dynamic(True)
#     resp.set_mass(0.5)
#     resp.set_respondable(True)
#     resp.set_model(True)
#     resp.set_parent(base)
#     resp.set_bullet_friction(1)

#     return resp
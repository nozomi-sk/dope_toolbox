import os
import random
import time
from glob import glob
import threading
import numpy as np
import nvisii
from numpy import deg2rad
from squaternion import Quaternion
import pybullet as p
from nvisii import vec3
import simplejson as json
import cv2
from typing import Union, Optional, Dict, Tuple
from pyquaternion import Quaternion as Pyq


class MakeDataset:
    _model_max_angles: Dict[str, Tuple[Optional[int], Optional[int], Optional[int]]]
    _latest_img: Optional[Union[np.ndarray, bool]]

    def __init__(self, root_path: str, objects_per_img: int = 20, preview=False, debug=False, overwrite=False):
        self._output_jpg = True
        self._root_path = root_path
        self._latest_img = None
        self._base_path = os.path.dirname(os.path.abspath(__file__))
        self._objects_per_img = objects_per_img
        self._spp = 150
        self._width = 400
        self._height = 400
        self._steps = 60
        self._hdr_paths = glob(os.path.join(self._base_path, "hdr", "*.hdr"))
        print("found %d hdr files" % len(self._hdr_paths))

        self.models = {}
        models_path = os.path.join(self._base_path, "models")
        enabled_models = [
            'ycb_002_master_chef_can',
            'ycb_004_sugar_box'
        ]
        self._model_max_angles = {
            # yaw, pitch, roll
            "ycb_002_master_chef_can": (0, None, None)
        }
        model_dirs = os.listdir(models_path)
        if len(enabled_models) > 0:
            model_dirs = [i for i in model_dirs if i in enabled_models]
        for model_dir in model_dirs:
            sub_dir_path = os.path.join(models_path, model_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            obj_paths = glob(os.path.join(sub_dir_path, "**/textured_fix.obj"))
            if len(obj_paths) == 1:
                print("found %s mesh" % model_dir)
                self.models[model_dir] = obj_paths[0]
            elif len(obj_paths) > 1:
                print("%s has multiple obj mesh files, ignore" % model_dir)
            else:
                print("%s has no obj mesh files, ignore" % model_dir)

        if len(self.models) == 0:
            raise RuntimeError("no available models found")

        self._camera_look_at = {
            'at': (0, 0, 0),
            'up': (0, 1, 1),
            'eye': (0, 0, 0.8)
        }
        self._pbt_client = None
        self._continue_event = threading.Event()
        self._continue_event.set()
        self._enable_preview = preview
        self._enable_debug = debug
        self._overwrite = overwrite

    @staticmethod
    def make_location():
        return (
            random.uniform(-0.25, 0.25),
            random.uniform(-0.25, 0.25),
            random.uniform(0.0, 0.3)
        )

    @staticmethod
    def make_rotation():
        new_rot = (
            random.uniform(-np.pi, np.pi),
            random.uniform(-np.pi, np.pi),
            random.uniform(-np.pi, np.pi),
        )
        q = Quaternion.from_euler(*new_rot)
        return q.x, q.y, q.z, q.w

    @staticmethod
    def is_valid_json(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return isinstance(data, dict)
        except ValueError:
            return False

    def _generate_one(self, output_path_without_ext: str):
        png_path = output_path_without_ext + ".png"
        json_path = output_path_without_ext + ".json"
        if self._output_jpg is True:
            img_path = output_path_without_ext + ".jpg"
        else:
            img_path = png_path

        if os.path.isfile(img_path) and \
                os.path.isfile(json_path) and \
                self.is_valid_json(json_path) and \
                self._overwrite is False:
            return

        nvisii.clear_all()
        p.resetSimulation()
        camera = nvisii.entity.create(
            name="camera",
            transform=nvisii.transform.create("camera"),
            camera=nvisii.camera.create(
                name="camera",
                aspect=float(self._width) / float(self._height)
            )
        )
        camera.get_transform().look_at(**self._camera_look_at)
        nvisii.set_camera_entity(camera)

        dome = nvisii.texture.create_from_file("dome", random.choice(self._hdr_paths))
        nvisii.set_dome_light_texture(dome)
        nvisii.set_dome_light_rotation(nvisii.angleAxis(deg2rad(random.random() * 720), vec3(0, 0, 1)))

        obj_model_map = {}
        pbt_obj_map = []
        for _ in range(self._objects_per_img):
            obj_class_name = random.choice(list(self.models.keys()))
            obj_path = self.models[obj_class_name]

            scene = nvisii.import_scene(file_path=obj_path)
            obj = scene.entities[0]
            assert isinstance(obj, nvisii.entity)
            obj_name = obj.get_name()
            obj_model_map[obj_name] = obj_class_name

            pose = self.make_location()
            obj.get_transform().set_position(pose)
            rot = self.make_rotation()
            obj.get_transform().set_rotation(rot)

            vertices = obj.get_mesh().get_vertices()

            obj_col_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices=vertices,
            )

            p.createMultiBody(
                baseCollisionShapeIndex=obj_col_id,
                basePosition=pose,
                baseOrientation=rot,
                baseMass=0.01
            )

            pbt_obj_map.append({
                'pbt': obj_col_id,
                'nvi': obj_name
            })

        self._run_physics_engine(pbt_obj_map)

        def round_4(yaw_pitch_roll):
            return tuple(map(lambda x: round(x, 4), yaw_pitch_roll))

        axes = [
            nvisii.vec3(0, 0, 1),
            nvisii.vec3(0, 1, 0),
            nvisii.vec3(1, 0, 0)
        ]
        for _map in pbt_obj_map:
            object_name = _map['nvi']
            model_name = obj_model_map[object_name]
            if model_name in self._model_max_angles:
                max_angles = self._model_max_angles[model_name]
                entity = nvisii.entity.get(object_name)
                trans = entity.get_transform()
                q = self.nvi_q_to_pyq(trans.get_rotation())
                angles = q.yaw_pitch_roll
                for index, max_angle in enumerate(max_angles):
                    current_angle = angles[index]
                    axis = axes[index]
                    if max_angle is not None:
                        trans.add_angle_axis(angle=max_angle - current_angle, axis=axis)

                new_angles = self.nvi_q_to_pyq(trans.get_rotation()).yaw_pitch_roll
                for index, max_angle in enumerate(max_angles):
                    current_angle = new_angles[index]
                    if max_angle is not None:
                        assert abs(max_angle - current_angle) < 0.01

        export_obj_names = []
        for _map in pbt_obj_map:
            obj_name = _map['nvi']
            export_obj_names.append(obj_name)
            _, dimensions_dict = self._add_cuboid(obj_name)

            model_name = obj_model_map[obj_name]
            dimension_path = os.path.join(self._root_path, model_name + ".json")
            if not os.path.isfile(dimension_path):
                with open(dimension_path, 'w+') as fp:
                    json.dump(dimensions_dict, fp, indent=4, sort_keys=True)

        self._export_json(json_path, export_obj_names, obj_model_map)
        print(json_path)

        nvisii.render_to_file(
            width=self._width,
            height=self._height,
            samples_per_pixel=self._spp,
            file_path=png_path
        )
        if img_path != png_path:
            image = cv2.imread(png_path)
            cv2.imwrite(img_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            os.unlink(png_path)

        print(img_path)
        self._latest_img = cv2.imread(img_path)

    @staticmethod
    def nvi_q_to_pyq(_nvi_q):
        return Pyq(w=_nvi_q.w, x=_nvi_q.x, y=_nvi_q.y, z=_nvi_q.z)

    @staticmethod
    def _get_cuboid_image_space(obj_name):
        cam_matrix = nvisii.entity.get('camera').get_transform().get_world_to_local_matrix()
        cam_proj_matrix = nvisii.entity.get('camera').get_camera().get_projection()

        points = []
        points_cam = []
        for i_t in range(9):
            trans = nvisii.transform.get(f"{obj_name}_cuboid_{i_t}")
            mat_trans = trans.get_local_to_world_matrix()
            pos_m = nvisii.vec4(
                mat_trans[3][0],
                mat_trans[3][1],
                mat_trans[3][2],
                1)

            p_cam = cam_matrix * pos_m

            p_image = cam_proj_matrix * (cam_matrix * pos_m)
            p_image = nvisii.vec2(p_image) / p_image.w
            p_image = p_image * nvisii.vec2(1, -1)
            p_image = (p_image + nvisii.vec2(1, 1)) * 0.5

            points.append([p_image[0], p_image[1]])
            points_cam.append([p_cam[0], p_cam[1], p_cam[2]])

        return points, points_cam

    def _export_json(self, save_path, obj_names, obj_model_map, visibility_use_percentage=False):
        camera_entity = nvisii.entity.get("camera")
        camera_trans = camera_entity.get_transform()
        # assume we only use the view camera
        cam_matrix = camera_trans.get_world_to_local_matrix()

        cam_matrix_export = []
        for row in cam_matrix:
            cam_matrix_export.append([row[0], row[1], row[2], row[3]])

        cam_world_location = camera_trans.get_position()
        cam_world_quaternion = camera_trans.get_rotation()

        cam_intrinsics = camera_entity.get_camera().get_intrinsic_matrix(self._width, self._height)
        dict_out = {
            "camera_data": {
                "width": self._width,
                'height': self._height,
                'camera_look_at':
                    {
                        'at': [
                            self._camera_look_at['at'][0],
                            self._camera_look_at['at'][1],
                            self._camera_look_at['at'][2],
                        ],
                        'eye': [
                            self._camera_look_at['eye'][0],
                            self._camera_look_at['eye'][1],
                            self._camera_look_at['eye'][2],
                        ],
                        'up': [
                            self._camera_look_at['up'][0],
                            self._camera_look_at['up'][1],
                            self._camera_look_at['up'][2],
                        ]
                    },
                'camera_view_matrix': cam_matrix_export,
                'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                'quaternion_world_xyzw': [
                    cam_world_quaternion[0],
                    cam_world_quaternion[1],
                    cam_world_quaternion[2],
                    cam_world_quaternion[3],
                ],
                'intrinsics': {
                    'fx': cam_intrinsics[0][0],
                    'fy': cam_intrinsics[1][1],
                    'cx': cam_intrinsics[2][0],
                    'cy': cam_intrinsics[2][1]
                }
            },
            "objects": []
        }

        # Segmentation id to export
        id_keys_map = nvisii.entity.get_name_to_id_map()

        for obj_name in obj_names:
            projected_key_points, _ = self._get_cuboid_image_space(obj_name)

            # put them in the image space.
            for i_p, _p in enumerate(projected_key_points):
                projected_key_points[i_p] = [_p[0] * self._width, _p[1] * self._height]

            # Get the location and rotation of the object in the camera frame

            trans = nvisii.entity.get(obj_name).get_transform()
            quaternion_xyzw = nvisii.inverse(cam_world_quaternion) * trans.get_rotation()

            object_world = nvisii.vec4(
                trans.get_position()[0],
                trans.get_position()[1],
                trans.get_position()[2],
                1
            )
            pos_camera_frame = cam_matrix * object_world

            # check if the object is visible
            bounding_box = [-1, -1, -1, -1]

            seg_mask = nvisii.render_data(
                width=self._width,
                height=self._height,
                start_frame=0,
                frame_count=1,
                bounce=0,
                options="entity_id",
            )
            seg_mask = np.array(seg_mask).reshape((self._width, self._height, 4))[:, :, 0]

            if visibility_use_percentage is True and int(id_keys_map[obj_name]) in np.unique(seg_mask.astype(int)):
                transforms_to_keep = {}

                for _name in id_keys_map.keys():
                    if 'camera' in _name.lower() or obj_name in _name:
                        continue
                    trans_to_keep = nvisii.entity.get(_name).get_transform()
                    transforms_to_keep[_name] = trans_to_keep
                    nvisii.entity.get(_name).clear_transform()

                # Percentage visibility through full segmentation mask.
                seg_unique_mask = nvisii.render_data(
                    width=self._width,
                    height=self._height,
                    start_frame=0,
                    frame_count=1,
                    bounce=0,
                    options="entity_id",
                )

                seg_unique_mask = np.array(seg_unique_mask).reshape((self._width, self._height, 4))[:, :, 0]

                values_segmentation = np.where(seg_mask == int(id_keys_map[obj_name]))[0]
                values_segmentation_full = np.where(seg_unique_mask == int(id_keys_map[obj_name]))[0]
                visibility = len(values_segmentation) / float(len(values_segmentation_full))

                # set back the objects from remove
                for entity_name in transforms_to_keep.keys():
                    nvisii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])
            else:
                if int(id_keys_map[obj_name]) in np.unique(seg_mask.astype(int)):
                    visibility = 1
                    y, x = np.where(seg_mask == int(id_keys_map[obj_name]))
                    bounding_box = [int(min(x)), int(max(x)), self._height - int(max(y)), self._height - int(min(y))]
                else:
                    visibility = 0

            object_class_name = obj_model_map[obj_name]
            dict_out['objects'].append({
                'class': object_class_name,
                'name': "%s_%d" % (object_class_name, round(time.time() * 1000)),
                'provenance': 'nvisii',
                'location': [
                    pos_camera_frame[0],
                    pos_camera_frame[1],
                    pos_camera_frame[2]
                ],
                'quaternion_xyzw': [
                    quaternion_xyzw[0],
                    quaternion_xyzw[1],
                    quaternion_xyzw[2],
                    quaternion_xyzw[3],
                ],
                'quaternion_xyzw_world': [
                    trans.get_rotation()[0],
                    trans.get_rotation()[1],
                    trans.get_rotation()[2],
                    trans.get_rotation()[3]
                ],
                'projected_cuboid': projected_key_points[0:8],
                'projected_cuboid_centroid': projected_key_points[8],
                # 'segmentation_id': id_keys_map[obj_name],
                'segmentation_id': 0,
                'visibility_image': visibility,
                'bounding_box': {
                    'top_left': [
                        bounding_box[0],
                        bounding_box[2],
                    ],
                    'bottom_right': [
                        bounding_box[1],
                        bounding_box[3],
                    ],
                },
            })

        with open(save_path, 'w+') as fp:
            json.dump(dict_out, fp, indent=4, sort_keys=True)
        return dict_out

    def _run_physics_engine(self, pbt_nvi_map):
        for i in range(self._steps):
            p.stepSimulation()

        for _map in pbt_nvi_map:
            pos, rot = p.getBasePositionAndOrientation(_map['pbt'])
            entity = nvisii.entity.get(_map['nvi'])
            entity.get_transform().set_position(pos)
            entity.get_transform().set_rotation(rot)

    @staticmethod
    def _add_cuboid(entity_name):
        obj = nvisii.entity.get(entity_name)
        min_obj = obj.get_mesh().get_min_aabb_corner()
        max_obj = obj.get_mesh().get_max_aabb_corner()
        centroid_obj = obj.get_mesh().get_aabb_center()
        dimensions_dict = {
            'width': max_obj[0] - min_obj[0],
            'height': max_obj[1] - min_obj[1],
            'length': max_obj[2] - min_obj[2]
        }
        cuboid1 = [
            vec3(max_obj[0], max_obj[1], max_obj[2]),
            vec3(min_obj[0], max_obj[1], max_obj[2]),
            vec3(max_obj[0], min_obj[1], max_obj[2]),
            vec3(max_obj[0], max_obj[1], min_obj[2]),
            vec3(min_obj[0], min_obj[1], max_obj[2]),
            vec3(max_obj[0], min_obj[1], min_obj[2]),
            vec3(min_obj[0], max_obj[1], min_obj[2]),
            vec3(min_obj[0], min_obj[1], min_obj[2]),
            vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]),
        ]

        cuboid2 = [
            cuboid1[2], cuboid1[0], cuboid1[3],
            cuboid1[5], cuboid1[4], cuboid1[1],
            cuboid1[6], cuboid1[7], cuboid1[-1],
            vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2])
        ]

        for i_p, p in enumerate(cuboid2):
            child_transform = nvisii.transform.create(f"{entity_name}_cuboid_{i_p}")
            child_transform.set_position(p)
            child_transform.set_scale(vec3(0.3))
            child_transform.set_parent(obj.get_transform())

        for i_v, v in enumerate(cuboid2):
            cuboid2[i_v] = [v[0], v[1], v[2]]

        return cuboid2, dimensions_dict

    def _preview(self):
        while True:
            if isinstance(self._latest_img, np.ndarray):
                cv2.imshow("preview", self._latest_img)
                if self._enable_debug:
                    self._continue_event.set()
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(20)
            elif self._latest_img is None:
                time.sleep(0.1)
            else:
                break

    def run(self, output_path, jobs):
        if self._enable_preview:
            preview_thread = threading.Thread(target=self._preview)
            preview_thread.daemon = True
            preview_thread.start()

        if not os.path.isdir(output_path):
            raise ValueError("%s is not valid" % output_path)

        nvisii.initialize(headless=True)
        nvisii.enable_denoiser()
        self._pbt_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)

        try:
            # [start,end]
            for output_name in jobs:
                self._continue_event.wait()
                self._generate_one(os.path.join(output_path, output_name))
                if self._enable_preview and self._enable_debug:
                    self._continue_event.clear()

            time.sleep(1)
        except KeyboardInterrupt:
            pass

        self._latest_img = False

        p.disconnect()
        cv2.destroyAllWindows()
        # let's clean up GPU resources
        nvisii.deinitialize()


if __name__ == '__main__':
    def main():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_per_img', default=20, dest="obj_per_img", type=int)
        parser.add_argument('--root', required=True, dest="root", type=str)
        parser.add_argument('--save', required=True, dest="save", type=str)
        parser.add_argument('--jobs', required=True, dest="jobs", type=str)
        parser.add_argument('--preview', default=0, dest="preview", type=int)
        parser.add_argument('--debug', default=0, dest="debug", type=int)
        parser.add_argument('--overwrite', default=0, dest="overwrite", type=int)

        args, _ = parser.parse_known_args()
        jobs = list(filter(lambda x: len(x) > 0, map(lambda x: str(x).strip(), str(args.jobs).split(','))))
        _m = MakeDataset(args.root, args.obj_per_img, preview=bool(args.preview), debug=bool(args.debug),
                         overwrite=bool(args.overwrite))
        _m.run(args.save, jobs)


    main()

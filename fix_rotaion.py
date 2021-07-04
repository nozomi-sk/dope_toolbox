import tempfile
import os
import random
import time
from glob import glob
import numpy as np
import nvisii
from numpy import deg2rad
from squaternion import Quaternion
import pybullet as p
from nvisii import vec3
import simplejson as json
import cv2
from typing import Union, Optional


class FixRotation:
    _latest_img: Optional[Union[np.ndarray, bool]]

    def __init__(self):
        self._base_path = os.path.dirname(os.path.abspath(__file__))
        self._spp = 200
        self._width = 400
        self._height = 400
        self._steps = 60
        self._objects_per_img = 3
        self._objs_vertices = {}
        self._hdr_paths = glob(os.path.join(self._base_path, "hdr", "*.hdr"))
        print("found %d hdr files" % len(self._hdr_paths))

        self.models = {
            'ycb_002_master_chef_can': os.path.join(
                self._base_path,
                "models/ycb_002_master_chef_can/meshes/textured_fix.obj"
            )
        }

        if len(self.models) == 0:
            raise RuntimeError("no available models found")

        # http://learnwebgl.brown37.net/07_cameras/camera_introduction.html
        self._camera_look_at = {
            'at': (0, 0, 0),
            'up': (0, -1, 0),
            'eye': (0, 0, 0.8)
        }
        self._pbt_client = None

    @staticmethod
    def make_location():
        return (
            random.uniform(-0.25, 0.25),
            random.uniform(-0.25, 0.25),
            random.uniform(-0.25, 0.25),
        )

    @staticmethod
    def make_rotation():
        new_rot = (
            random.uniform(0, 90),  # Roll
            random.uniform(0, 90),  # Pitch
            0  # Yaw
        )
        q = Quaternion.from_euler(*new_rot, degrees=True)
        return q.x, q.y, q.z, q.w

    @staticmethod
    def is_valid_json(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return isinstance(data, dict)
        except ValueError:
            return False

    def _cache_vertices(self):
        start_time = time.time()
        for obj_class_name, obj_path in self.models.items():
            nvisii.clear_all()
            scene = nvisii.import_scene(file_path=obj_path)
            obj = scene.entities[0]
            assert isinstance(obj, nvisii.entity)
            self._objs_vertices[obj_class_name] = obj.get_mesh().get_vertices()
        nvisii.clear_all()
        print("cache time: %.4f" % (time.time() - start_time))

    def _generate_one(self):
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

        start_time = time.time()
        obj_model_map = {}
        tries = 0

        while len(obj_model_map.keys()) < self._objects_per_img:
            tries += 1
            obj_class_name = random.choice(list(self.models.keys()))
            obj_path = self.models[obj_class_name]
            pose = self.make_location()
            rot = self.make_rotation()
            vertices = self._objs_vertices[obj_class_name]
            pbt_object_id = p.createCollisionShape(
                p.GEOM_MESH,
                vertices=vertices,
            )
            body_id = p.createMultiBody(
                baseCollisionShapeIndex=pbt_object_id,
                basePosition=pose,
                baseOrientation=rot,
                baseMass=0.01
            )
            if not self._has_collision(body_id):
                scene = nvisii.import_scene(file_path=obj_path)
                obj = scene.entities[0]
                assert isinstance(obj, nvisii.entity)
                obj.get_transform().set_position(pose)
                obj.get_transform().set_rotation(rot)
                obj_name = obj.get_name()
                obj_model_map[obj_name] = obj_class_name
            else:
                p.removeBody(body_id)

        print("tries: %d, time: %.4f" % (tries, (time.time() - start_time)))
        export_obj_names = list(obj_model_map.keys())
        for export_obj_name in export_obj_names:
            _, dimensions_dict = self._add_cuboid(export_obj_name)
            print(obj_model_map[export_obj_name], dimensions_dict)

        ori_img_data = self.nvisii_to_cv()
        assert isinstance(ori_img_data, np.ndarray)
        print(ori_img_data.shape)
        cv2.imshow("ori", ori_img_data)
        point_img_data = ori_img_data.copy()
        self.draw_points(export_obj_names, 'cuboid', point_img_data)
        cv2.imshow("points", point_img_data)
        cv2.waitKey(0)

    def draw_points(self, obj_names, transform_name, image_mat):
        def hex_to_rgb(hex_str):
            h = hex_str.lstrip('#')
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))[::-1]

        colors = [
            '#524034',
            '#518f14',
            '#61e6f3',
            '#89367b',
            '#96d03d',
            '#897a98',
            '#07f79c',
            '#5175a0',
            '#e3d586'
        ]
        colors = list(map(lambda x: hex_to_rgb(x), colors))

        for obj_name in obj_names:
            projected_key_points, _ = self._get_cuboid_image_space(obj_name, transform_name)

            points = []
            # put them in the image space.
            for i_p, _p in enumerate(projected_key_points):
                x, y = _p[0] * self._width, _p[1] * self._height
                points.append((round(x), round(y)))

            def draw_line(_p1, _p2, color=(255, 255, 255)):
                cv2.line(image_mat, _p1, _p2, color, thickness=1)

            # draw front
            draw_line(points[0], points[1], (0, 0, 255))
            draw_line(points[1], points[2], (0, 0, 255))
            draw_line(points[2], points[3], (0, 0, 255))
            draw_line(points[3], points[0], (0, 0, 255))
            draw_line(points[0], points[2], (0, 0, 255))
            draw_line(points[1], points[3], (0, 0, 255))

            # draw back
            draw_line(points[4], points[5])
            draw_line(points[5], points[6])
            draw_line(points[6], points[7])
            draw_line(points[7], points[4])

            # draw sides
            draw_line(points[1], points[5])
            draw_line(points[2], points[6])
            draw_line(points[0], points[4])
            draw_line(points[3], points[7])

            for i, point in enumerate(points[:8]):
                cv2.circle(image_mat, point, 5, colors[i], thickness=-1)

    def nvisii_to_cv(self):
        fd, save_path = tempfile.mkstemp(prefix="lasr_dope_", suffix=".png")
        nvisii.render_to_file(
            width=self._width,
            height=self._height,
            samples_per_pixel=self._spp,
            file_path=save_path
        )
        os.close(fd)
        img_data = cv2.imread(save_path)
        os.unlink(save_path)
        return img_data

    @staticmethod
    def _get_cuboid_image_space(obj_name, transform_name="cuboid"):
        cam_matrix = nvisii.entity.get('camera').get_transform().get_world_to_local_matrix()
        cam_proj_matrix = nvisii.entity.get('camera').get_camera().get_projection()

        points = []
        points_cam = []
        for i_t in range(9):
            trans = nvisii.transform.get(f"{obj_name}_{transform_name}_{i_t}")
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

    @staticmethod
    def _has_collision(body_id):
        p.stepSimulation()
        contact_points = p.getContactPoints(bodyA=body_id)
        return len(contact_points) > 0

    @staticmethod
    def _add_cuboid(entity_name, transform_name="cuboid"):
        obj = nvisii.entity.get(entity_name)
        min_obj = obj.get_mesh().get_min_aabb_corner()
        max_obj = obj.get_mesh().get_max_aabb_corner()
        centroid_obj = obj.get_mesh().get_aabb_center()
        dimensions_dict = {
            'x': max_obj[0] - min_obj[0],
            'y': max_obj[1] - min_obj[1],
            'z': max_obj[2] - min_obj[2]
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
            cuboid1[2], cuboid1[4], cuboid1[1],
            cuboid1[0], cuboid1[5], cuboid1[7],
            cuboid1[6], cuboid1[3], cuboid1[-1],
            vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2])
        ]

        for i_p, p in enumerate(cuboid2):
            child_transform = nvisii.transform.create(f"{entity_name}_{transform_name}_{i_p}")
            child_transform.set_position(p)
            child_transform.set_scale(vec3(0.3))
            child_transform.set_parent(obj.get_transform())

        for i_v, v in enumerate(cuboid2):
            cuboid2[i_v] = [v[0], v[1], v[2]]

        return cuboid2, dimensions_dict

    def run(self):
        nvisii.initialize(headless=True)
        nvisii.enable_denoiser()
        self._pbt_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        self._cache_vertices()
        try:
            while True:
                self._generate_one()
        except KeyboardInterrupt:
            pass

        p.disconnect()
        cv2.destroyAllWindows()
        # let's clean up GPU resources
        nvisii.deinitialize()


if __name__ == '__main__':
    def main():
        _m = FixRotation()
        _m.run()


    main()

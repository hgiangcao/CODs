from pyrep.robots.end_effectors.baxter_suction_cup import BaxterSuctionCup
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object


class SteroidBaxterSuctionCup(BaxterSuctionCup):

    def __init__(self, count: int = 0):
        super().__init__(count=count)

        suction_sensor_root = Dummy('BaxterSuctionCup_sensor_root')

        sensors = suction_sensor_root.get_objects_in_tree()

        self.sensors = [obj for obj in sensors if 'sensor' in obj.get_name().lower()]


    def grasp(self, obj: Object) -> bool:
        """Attach the object to the suction cup if it is detected.

        EDIT: attach only if all sensors detect the object

        Note: The does not move the object up to the suction cup. Therefore, the
        proximity sensor should have a short range in order for the suction
        grasp to look realistic.

        :param obj: The object to grasp if detected.
        :return: True if the object was detected/grasped.
        """
        # detected = self._proximity_sensor.is_detected(obj)
        detected = True
        for sensor in self.sensors:
            if not sensor.is_detected(obj):
                detected = False
                break

        # Check if detected and that we are not already grasping it.
        if detected and obj not in self._grasped_objects:
            self._grasped_objects.append(obj)
            self._old_parents.append(obj.get_parent())  # type: ignore
            obj.set_parent(self._attach_point, keep_in_place=True)
            obj.set_model_dynamic(False)
        return detected
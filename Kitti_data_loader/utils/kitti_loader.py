class KITTILoader:
    def __init__(self, base_path: str = r"Data/KITTI/000", frame_id=0, time_unit="ns"):
        """
        time_unit: 'ns' for nanoseconds, 'ms' for milliseconds
        """
        import os
        from Kitti_data_loader.utils.image_loader import load_image_list
        from Kitti_data_loader.utils.oxts_parser import load_oxts_data

        self.base_path = base_path
        self.image_dir = os.path.join(base_path, "image_00/data")
        self.image_timestamps = os.path.join(base_path, "image_00/timestamps.txt")
        self.oxts_dir = os.path.join(base_path, "oxts/data")
        self.oxts_timestamps = os.path.join(base_path, "oxts/timestamps.txt")

        self.time_unit = time_unit

        # Load all images and OXTS
        self.images = load_image_list(self.image_dir, self.image_timestamps)
        self.oxts = load_oxts_data(self.oxts_dir, self.oxts_timestamps)

        # Convert relative time to ns or ms
        self._convert_rel_time()

        self.frame_id = frame_id

    def _convert_rel_time(self):
        """Convert relative times to integer ns or ms"""
        factor = 1_000_000_000 if self.time_unit == "ns" else 1_000  # ns or ms
        # Images
        self.images = [
            (date, time, int(rel * factor), img) for date, time, rel, img in self.images
        ]
        # OXTS
        self.oxts = [
            (date, time, int(rel * factor), data) for date, time, rel, data in self.oxts
        ]

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        self.frame_id = 0
        return self

    def __next__(self):
        if self.frame_id >= len(self.images):
            raise StopIteration
        data = self.get_next_data()
        return data

    def get_data(self, idx):
        """Return (img, img_rel, oxts_dict, oxts_rel)"""
        _, _, img_rel, img = self.images[idx]
        _, _, oxts_rel, oxts_dict = self.oxts[idx]
        return img, img_rel, oxts_dict, oxts_rel

    def get_next_data(self):
        """Return (img, img_rel, oxts_dict, oxts_rel)"""
        if self.frame_id >= self.__len__():
            return None
        _, _, img_rel, img = self.images[self.frame_id]
        _, _, oxts_rel, oxts_dict = self.oxts[self.frame_id]
        self.frame_id = self.frame_id + 1
        return img, img_rel, oxts_dict, oxts_rel

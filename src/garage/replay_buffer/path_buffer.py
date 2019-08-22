"""A replay buffer that efficiently stores and can sample whole paths."""
import collections

import numpy as np


class PathBuffer:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.
    Args:
        capacity_in_steps (int): total memory allocated for the buffer
    """

    def __init__(self, capacity_in_transitions):
        self._capacity = capacity_in_transitions
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path(dict): A dict of array of shape (path_len, flat_dim)
        """
        for key, buf_arr in self._buffer.items():
            path_array = path.get(key, None)
            if path_array is None:
                raise ValueError('Key {} missing from path.'.format(key))
            elif (len(path_array.shape) != 2
                  or path_array.shape[1] != buf_arr.shape[1]):
                raise ValueError('Array {} has wrong shape.'.format(key))
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            buf_arr = self._get_or_allocate_key(key, array)
            # numpy doesn't special case range indexing, so it's very slow.
            # Slice manually instead, which is faster than any other method.
            # pylint: disable=invalid-slice-index
            buf_arr[first_seg.start:first_seg.stop] = array[:len(first_seg)]
            buf_arr[second_seg.start:second_seg.stop] = array[len(first_seg):]
        if second_seg.stop:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + path_len)

    def sample_path(self):
        """Sample a single path from the buffer.

        Returns:
            A dict of arrays of shape (path_len, flat_dim)

        """
        path_idx = np.random.randint(len(self._path_segments))
        first_seg, second_seg = self._path_segments[path_idx]
        first_seg_indices = np.arange(first_seg.start, first_seg.stop)
        second_seg_indices = np.arange(second_seg.start, second_seg.stop)
        indices = np.concatenate([first_seg_indices, second_seg_indices])
        path = {key: buf_arr[indices] for key, buf_arr in self._buffer.items()}
        return path

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Returns:
            A dict of arrays of shape (batch_size, flat_dim)

        """
        idx = np.random.randint(self._transitions_stored, size=batch_size)
        return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored."""
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity, array.shape[1]), array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    @staticmethod
    def _get_path_length(path):
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Returns:
            True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop >= second.start

import tensorflow as tf
import collections
import six

# My own clip by value which could input a list of tensors
def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")
    t_list = list(t_list)
        
    with tf.name_scope(name or "clip_by_value") as name:
    #with ops.name_scope(name, "clip_by_global_norm",
    #              t_list + [clip_norm]) as name:
    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(t_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped
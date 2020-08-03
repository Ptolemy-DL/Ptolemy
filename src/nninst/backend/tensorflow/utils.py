from contextlib import contextmanager

import tensorflow as tf

from nninst import mode
from nninst.utils.fs import abspath

__all__ = ["initialize_uninitialized_vars", "new_session_config", "rename_checkpoint"]


def initialize_uninitialized_vars(sess):
    from itertools import compress

    global_vars = tf.global_variables()
    is_not_initialized = sess.run(
        [~(tf.is_variable_initialized(var)) for var in global_vars]
    )
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


@contextmanager
def restore_scope(sess: tf.Session, checkpoint_path: str):
    old_variables = tf.global_variables()
    yield
    current_variables = tf.global_variables()
    new_variables = list(set(current_variables) - set(old_variables))
    if len(new_variables) != 0:
        train_saver = tf.train.Saver(var_list=new_variables)
        train_saver.restore(sess, checkpoint_path)


def new_session_config(parallel: int = 1):
    session_config = tf.ConfigProto(allow_soft_placement=True)
    if not mode.is_debug():
        session_config.intra_op_parallelism_threads = parallel
        session_config.inter_op_parallelism_threads = parallel
    session_config.gpu_options.allow_growth = True
    return session_config


def rename_checkpoint(
    checkpoint_dir, replace_from, replace_to, add_prefix=None, dry_run=False
):
    checkpoint_dir = abspath(checkpoint_dir)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print("%s would be renamed to %s." % (var_name, new_name))
            else:
                print("Renaming %s to %s." % (var_name, new_name))
                # Rename the variable
                tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)

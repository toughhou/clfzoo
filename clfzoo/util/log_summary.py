import tensorflow as tf
import os


def summary_op(loss, acc):
    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("acc", acc)

    summary_op = tf.summary.merge([loss_summary, acc_summary])

    return summary_op


def summary_add(out_dir, sess, summaries, step):
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    train_summary_writer.add_summary(summaries, step)

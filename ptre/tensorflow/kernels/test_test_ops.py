import tensorflow as tf
test_ops_module = tf.load_op_library('./test_ops.so')
print dir(test_ops_module)
print "calling is_incoming"
incoming = test_ops_module.is_incoming()
print "type of incoming: "
print type(incoming)
#incoming.eval()
print incoming.numpy()

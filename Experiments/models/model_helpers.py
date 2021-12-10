from tensorflow.keras.layers import Input,Concatenate
from tensorflow.keras import Model

def parallelize(m1,m2):
    inp = Input(m1.input_shape[1:])
    outp_1 = m1(inp)
    outp_2 = m2(inp)
    outp_con = Concatenate()([outp_1,outp_2])
    combined_model = Model(inp,outp_con)
    return combined_model